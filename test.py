import argparse
from dataloader import MRIDataset
from model.resunet3d import ResUNet3D
from model.r2unet3d import R2UNet3D
from model.unet3d import UNet3D
from torch.utils.data import DataLoader
from torch.nn import DataParallel
import torch
import logging
import torchvision.transforms as T

from utils.evaluate import evaluate
from utils.utils import load_checkpoint



def get_args():
	# Test settings
	parser = argparse.ArgumentParser(description='Evaluate using test loader')
	parser.add_argument('--network', '-u', default='Unet3D', help='Specify the network (Unet3D / ResidualUnet3D / R2Unet3D)')
	parser.add_argument('--model', '-m', default='model.pt', metavar='FILE', help='Specify the paht to the file in which the model is stored (model.pt)')
	parser.add_argument('--batch-size', type=int, default=1, metavar='N',help='input batch size for testing (default: 1)')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA testing (default: False)')
	parser.add_argument('--mask-threshold', '-t', type=float, default=0.5, help='Minimum probability value to consider a mask pixel white (default: 0.5)')

	return parser.parse_args()


def main():

	args = get_args()
	logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

	use_cuda = not args.no_cuda and torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")

	# Specify network
	if args.network.casefold() == "unet3d":
		model = UNet3D(in_channels=1, out_channels=1).to(device)
	elif args.network.casefold() == "residualunet3d":
		model = ResUNet3D(in_channels=1, out_channels=1).to(device)
	elif args.network.casefold() == "r2unet3d":
		model = R2UNet3D(in_channels=1, out_channels=1).to(device)

	# If using multiple gpu
	if torch.cuda.device_count() > 1 and use_cuda:
		model = DataParallel(model)

	# Load trained model
	load_checkpoint(args.model, model ,device=device)
		
	test_kwargs = {'batch_size': args.batch_size}

	if use_cuda:
		cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': False}
		test_kwargs.update(cuda_kwargs)

	# Data Loading
	testdataset = MRIDataset(train=False, transform=T.ToTensor(), elastic=False)
	test_loader = DataLoader(dataset=testdataset, **test_kwargs)

	logging.info(f'''Starting testing:
        Network:         {args.network}
        Batch size:      {args.batch_size}
        Testing size:   {len(test_loader)}
        Device:          {device.type}
		Mask Threshold:  {args.mask_threshold}
    ''')

	evaluate(model, test_loader, device, args.mask_threshold, show_stat=True, plot=True, ttest=True)


if __name__ == '__main__':
    main()
