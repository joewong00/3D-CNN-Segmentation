import argparse
import matplotlib.pyplot as plt
from dataloader import MRIDataset
from residual3dunet.model import ResidualUNet3D
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel
import numpy as np
import torch
import torchvision.transforms as T
from segmentation_statistics import SegmentationStatistics
from utils import compute_average
import os

def prepare_plot(features, labels, preds, depth):
	# Visualize Single Image Data
	f, axarr = plt.subplots(depth,3,figsize=(50,50))

	# Convert to cpu
	features = features.cpu()
	labels = labels.cpu()
	preds = preds.cpu()

	for i in range(depth):
		axarr[i,0].imshow(features[0,0,i,:,:],cmap='gray')
		axarr[i,1].imshow(preds[0,0,i,:,:],cmap='gray')
		axarr[i,2].imshow(labels[0,0,i,:,:],cmap='gray')
		plt.axis('off')

	plt.show()
	plt.savefig('output.png')

def predict(model, device, loader):


	stats = []
	model.eval()
	# Disable grad
	with torch.no_grad():

		for batch_idx, (data, target) in enumerate(loader):
			data, target = data.float().to(device), target.float().to(device)
			output = model(data)

			preds = (output > 0.5).float()

			# Convert to numpy boolean
			preds = preds.numpy()
			target = target.numpy()	
			preds = preds.astype(bool)
			target = target.astype(bool)

			# batch, channel, depth, width, height = preds.shape

			stat = SegmentationStatistics(preds[0,0,:,:,:], target[0,0,:,:,:], (3,2,1))
			stats.append(stat)
			

			# num_correct += (preds == target).sum()
			# num_pixels += torch.numel(preds)
			# dice_score += dice_coefficient(preds, target)
			# jaccard += iou(preds, target)

			# print("Test set "+str(batch_idx + 1))
			# print("Dice score: "+str(dice_coefficient(preds, target).item()))
			# print("IOU: "+str(iou(preds, target).item()))
			# print()

		# Average
		print("All:")
		print(compute_average(stats, dataframe=True))

		# HC
		print("\nHealthy Control:")
		print(compute_average(stats,0,25,dataframe=True))

		# CKD
		print("\nChronic Kidney Disease:")
		print(compute_average(stats,25,None,dataframe=True))

		# print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
		# print(f"Dice score: {dice_score/len(loader)}")
		# print(f"IOU: {jaccard/len(loader)}")
		# prepare_plot(data, target, preds, depth)


def main():
	# Testing settings
	parser = argparse.ArgumentParser(description='PyTorch 3D Segmentation')
	parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 64)')
				
	parser.add_argument('--model', type=int, default=1, metavar='M',
                        help='model number (default: 1)')

	parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA testing')

	parser.add_argument('--multi-gpu', action='store_true', default=False,
                        help='use multiple gpu for training')

	args = parser.parse_args()
	use_cuda = not args.no_cuda and torch.cuda.is_available()

	device = torch.device("cuda" if use_cuda else "cpu")

	model = ResidualUNet3D(in_channels=1, out_channels=1, testing=True).to(device)

	# If using multiple gpu
	if args.multi_gpu:
		model = DataParallel(model)

	model.load_state_dict(torch.load(f"model{args.model}.pt", map_location=device))
		
	test_kwargs = {'batch_size': args.batch_size}

	if use_cuda:
		cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
		test_kwargs.update(cuda_kwargs)

	testdataset = MRIDataset(train=False, transform=T.ToTensor())
	test_loader = DataLoader(dataset=testdataset, **test_kwargs)

	predict(model, device, test_loader)


if __name__ == '__main__':
    main()