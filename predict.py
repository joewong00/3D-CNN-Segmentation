import argparse
import torch
import logging
import os
import nibabel as nib

# from residual3dunet.model import UNet3D, ResidualUNet3D
from model.resunet3d import ResUNet3D
from model.r2unet3d import R2UNet3D
from model.unet3d import UNet3D
from torch.nn import DataParallel
from utils.segmentation_statistics import SegmentationStatistics
from utils.utils import load_checkpoint, read_data_as_numpy, numpy_to_nii, visualize2d, plot_sidebyside, plot_overlapped, preprocess, predict


def get_args():
	# Test settings
	parser = argparse.ArgumentParser(description='Predict masks from input images')
	parser.add_argument('--network', '-u', default='Unet3D', help='Specify the network (Unet3D / ResidualUnet3D)')
	parser.add_argument('--model', '-m', default='model.pt', metavar='FILE', help='Specify the path to the file in which the model is stored (default:model.pt)')
	parser.add_argument('--input', '-i', metavar='INPUT', help='Path to the image file (format: nii.gz)', required=True)
	parser.add_argument('--mask', '-l', metavar='INPUT', default=None, help='Path to the ground truth of the input image (if_available) (default:None)')
	parser.add_argument('--viz', '-v', action='store_true',  default=True, help='Visualize the output (default:True)')
	parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA testing (default: False)')
	parser.add_argument('--mask-threshold', '-t', type=float, default=0.5, help='Minimum probability value to consider a mask pixel white (default: 0.5)')

	return parser.parse_args()

def main():
	
	args = get_args()
	logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
	filename = os.path.basename(args.input)
	use_cuda = not args.no_cuda and torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")

	assert args.network.casefold() in ("unet3d", "residualunet3d","r2unet3d"), 'Network must be either (Unet3D / ResidualUnet3D)'

	# Specify network
	if args.network.casefold() == "unet3d":
		model = UNet3D(in_channels=1, out_channels=1, testing=True).to(device)

	elif args.network.casefold() == "residualunet3d":
		model = ResUNet3D(in_channels=1, out_channels=1, testing=True).to(device)

	else:
		model = R2UNet3D(in_channels=1, out_channels=1, testing=True).to(device)

	# If using multiple gpu
	if torch.cuda.device_count() > 1 and use_cuda:
		model = DataParallel(model)

	logging.info(f'Loading model {args.model}')
	logging.info(f'Using device {device}')

	# Loading trained model
	load_checkpoint(args.model, model ,device=device)

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
		plot_sidebyside(data, prediction, target)

		prediction = prediction.astype(bool)
		target = target.astype(bool)

		stat = SegmentationStatistics(prediction, target, (3,2,1))
		stat.print_table()

if __name__ == '__main__':
    main()