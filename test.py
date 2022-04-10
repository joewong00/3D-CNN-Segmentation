import argparse
from dataloader import MRIDataset
from residual3dunet.model import ResidualUNet3D, UNet3D
from torch.utils.data import DataLoader
from torch.nn import DataParallel
import torch
import torchvision.transforms as T

from utils.evaluate import evaluate
from utils.utils import load_checkpoint



def get_args():
	# Test settings
	parser = argparse.ArgumentParser(description='Evaluate using test loader')
	parser.add_argument('--network', '-u', default='Unet3D', help='Specify the network (Unet3D / ResidualUnet3D)')
	parser.add_argument('--model', '-m', default='model.pt', metavar='FILE', help='Specify the paht to the file in which the model is stored (model.pt)')
	parser.add_argument('--batch-size', type=int, default=1, metavar='N',help='input batch size for testing (default: 1)')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA testing (default: False)')
	parser.add_argument('--mask-threshold', '-t', type=float, default=0.5, help='Minimum probability value to consider a mask pixel white (default: 0.5)')

	return parser.parse_args()


def main():

	args = get_args()

	use_cuda = not args.no_cuda and torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")

	# Specify network
	if args.network.casefold() == "unet3d":
		model = UNet3D(in_channels=1, out_channels=1, testing=True).to(device)

	else:
		model = ResidualUNet3D(in_channels=1, out_channels=1, testing=True).to(device)

	# If using multiple gpu
	if torch.cuda.device_count() > 1 and use_cuda:
		model = DataParallel(model)

	load_checkpoint(args.model, model ,device=device)
		
	test_kwargs = {'batch_size': args.batch_size}

	if use_cuda:
		cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
		test_kwargs.update(cuda_kwargs)

	testdataset = MRIDataset(train=False, transform=T.ToTensor())
	test_loader = DataLoader(dataset=testdataset, **test_kwargs)

	evaluate(model, test_loader, device, args.mask_threshold, show_stat=True)


if __name__ == '__main__':
    main()
