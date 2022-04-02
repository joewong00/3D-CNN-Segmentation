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
from utils import compute_average, prepare_plot
import os


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
			preds = preds.cpu().numpy()
			target = target.cpu().numpy()	
			preds = preds.astype(bool)
			target = target.astype(bool)

			# batch, channel, depth, width, height = preds.shape

			stat = SegmentationStatistics(preds[0,0,:,:,:], target[0,0,:,:,:], (3,2,1))
			stats.append(stat.to_dict())

		# Average
		print("All:")
		print(compute_average(stats, dataframe=True))

		# HC
		print("\nHealthy Control:")
		print(compute_average(stats,0,25,dataframe=True))

		# CKD
		print("\nChronic Kidney Disease:")
		print(compute_average(stats,25,None,dataframe=True))


def main():
	# Testing settings
	parser = argparse.ArgumentParser(description='PyTorch 3D Segmentation')
	parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 64)')
				
	parser.add_argument('--model', type=int, default=1, metavar='M',
                        help='model number (default: 1)')

	parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA testing')

	args = parser.parse_args()
	use_cuda = not args.no_cuda and torch.cuda.is_available()

	device = torch.device("cuda" if use_cuda else "cpu")

	model = ResidualUNet3D(in_channels=1, out_channels=1, testing=True).to(device)

	# If using multiple gpu
	if torch.cuda.device_count() > 1 and use_cuda:
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