import argparse
import torch
import logging
import os
import nibabel as nib

from residual3dunet.model import UNet3D
from residual3dunet.res3dunetmodel import ResidualUNet3D
from torch.nn import DataParallel
from utils.segmentation_statistics import SegmentationStatistics
from utils.utils import load_checkpoint, read_data_as_numpy, add_channel, to_depth_first, numpy_to_nii, visualize2d, to_depth_last, plot_overlapped, preprocess


def predict(model,input,threshold,device):

	model.eval()

	input = to_depth_first(input)

	if len(input.shape) == 3:
		input = add_channel(input)

	# Add batch dimension
	input = input.unsqueeze(0)
	input = input.to(device=device, dtype=torch.float32)

	# Disable grad
	with torch.no_grad():

		output = model(input)
		preds = (output > threshold).float()

		# Squeeze channel and batch dimension
		preds = torch.squeeze(preds)

		# Convert to numpy
		preds = preds.cpu().numpy()

	return preds

def get_args():
	# Test settings
	parser = argparse.ArgumentParser(description='Predict masks from input images')
	parser.add_argument('--network', '-u', default='Unet3D', help='Specify the network (Unet3D / ResidualUnet3D)')
	parser.add_argument('--model', '-m', default='model.pt', metavar='FILE', help='Specify the file in which the model is stored')
	parser.add_argument('--input', '-i', metavar='INPUT', help='Path to the image file (format: nii.gz)', required=True)
	parser.add_argument('--mask', '-l', metavar='INPUT', default=None, help='Path to the ground truth of the input image (if_available)')
	parser.add_argument('--viz', '-v', action='store_true', help='Visualize the output')
	parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA testing')
	parser.add_argument('--mask-threshold', '-t', type=float, default=0.5, help='Minimum probability value to consider a mask pixel white')

	return parser.parse_args()

def main():
	
	args = get_args()
	filename = os.path.basename(args.input)
	use_cuda = not args.no_cuda and torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")

	assert args.network.casefold() == "unet3d" or args.network.casefold() == "residualunet3d", 'Network must be either (Unet3D / ResidualUnet3D)'

	# Specify network
	if args.network.casefold() == "unet3d":
		model = UNet3D(in_channels=1, out_channels=1, testing=True).to(device)

	else:
		model = ResidualUNet3D(in_channels=1, out_channels=1, testing=True).to(device)

	logging.info(f'Loading model {args.model}')
	logging.info(f'Using device {device}')

	# If using multiple gpu
	if torch.cuda.device_count() > 1 and use_cuda:
		model = DataParallel(model)

	load_checkpoint(args.model, model ,device=device)
	# model.load_state_dict(torch.load(args.model, map_location=device))

	logging.info('Model loaded!')
	logging.info(f'\nPredicting image {filename} ...')

	data = preprocess(read_data_as_numpy(args.input), rotate=True, to_tensor=False)

	prediction = predict(model, data, args.mask_threshold, device)

	if not args.no_save:
		# Save prediction mask as nii.gz at output dir

		if not os.path.exists('output'):
			os.mkdir('output')

		image_data = numpy_to_nii(prediction)
		nib.save(image_data, f"output/Mask_{filename}")

		logging.info(f'\nMask saved to output/Mask_{filename}')

	if args.viz:
		visualize2d(prediction)

	# Evaluation statistics
	if args.mask is not None:
		target = preprocess(read_data_as_numpy(args.mask),rotate=True, to_tensor=False)

		plot_overlapped(data, prediction, target)

		prediction = prediction.astype(bool)
		target = target.astype(bool)

		stat = SegmentationStatistics(prediction, target, (3,2,1))
		stat.print_table()

if __name__ == '__main__':
    main()