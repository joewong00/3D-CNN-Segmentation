from torch.optim import Adam
from dataloader import MRIDataset
from evaluate import evaluate
from residual3dunet.model import ResidualUNet3D, UNet3D
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
from torch.nn import DataParallel
from utils import load_checkpoint, plot_train_loss, save_model
from lossfunction import DiceBCELoss, DiceLoss, IoULoss, FocalLoss, FocalTverskyLoss, TverskyLoss

import os
import torch
import argparse
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.transforms as T
import logging


# Training
def train(args, model, device, train_loader, optimizer, epoch, criterion):
    costs = []
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        assert data.shape[1] == model.in_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {data.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

        data, target = data.float().to(device), target.float().to(device)
        
        output = model(data)
        
        optimizer.zero_grad()
        cost = criterion(output, target)
        cost.backward()
        optimizer.step()

        costs.append(cost.item())

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), cost.item()))
            if args.dry_run:
                break

    avgcost = sum(costs)/len(costs)

    return avgcost


def test(model, device, test_loader, epoch, loss):
    costs = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.float().to(device), target.float().to(device)
            output = model(data)

            cost = loss(output, target)

            costs.append(cost.item())

            print('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(test_loader.dataset),
                100. * batch_idx / len(test_loader), cost.item()))

        avgcost = sum(costs)/len(costs)
    
    return avgcost


def get_args():
    # Train settings
    parser = argparse.ArgumentParser(description='PyTorch 3D Segmentation')
    parser.add_argument('--network', '-u', default='Unet3D', help='Specify the network (Unet3D / ResidualUnet3D)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=3, metavar='N',help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=2.5e-4, metavar='LR', help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.1, metavar='M',help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,help='For Saving the current Model')
    parser.add_argument('--checkpoint', '-c', metavar='FILE', help='Specify the path to the model')

    return parser.parse_args()


def main():

    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size, 'shuffle': True}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)


    assert args.network.casefold() == "unet3d" or args.network.casefold() == "residualunet3d", 'Network must be either (Unet3D / ResidualUnet3D)'

	# Specify network
    if args.network.casefold() == "unet3d":
        model = UNet3D(in_channels=1, out_channels=1).to(device)

    else:
        model = ResidualUNet3D(in_channels=1, out_channels=1).to(device)

    # If using multiple gpu
    if torch.cuda.device_count() > 1 and use_cuda:
        model = DataParallel(model)

    # If load checkpoint
    if args.checkpoint:
        load_checkpoint(args.checkpoint, model, device=device)

    logging.info(f'Network:\n'
                f'\t{model.in_channels} input channels\n'
                f'\t{model.out_channels} output channels (classes)\n')

    # Data Loading
    transformation = T.Compose([T.ToTensor(),
                    T.Normalize(),
                    T.RandomHorizontalFlip(),
                    T.RandomRotation(90),
                    T.RandomCrop((240,240), padding=50, pad_if_needed=True)])

    traindataset = MRIDataset(train=True, transform=transformation, elastic=True)
    # testdataset = MRIDataset(train=False, transform=T.ToTensor())

    train_set, val_set = random_split(traindataset, [int(len(traindataset)*0.9),int(len(traindataset)*0.1)])

    train_loader = DataLoader(dataset=train_set, **train_kwargs)
    val_loader = DataLoader(dataset=val_set, **train_kwargs)
    # test_loader = DataLoader(dataset=testdataset, **test_kwargs)
 
    # Hyperparameters
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=50, gamma=args.gamma)
    loss = DiceBCELoss()

    # Validation Loss
    minvalidation = 1
    loss_train = []
    loss_val = []
    min_dice = 1

    logging.info(f'''Starting training:
        Network:         {args.network}
        Epochs:          {args.epochs}
        Batch size:      {args.batch_size}
        Learning rate:   {args.lr}
        Training size:   {len(train_loader)}
        Validation size: {len(val_loader)}
        Device:          {device.type}
    ''')

    # Training process
    for epoch in range(1, args.epochs + 1):

        # Training
        trainloss = train(args, model, device, train_loader, optimizer, epoch, loss)
        valloss = evaluate(model, val_loader, device, loss)

        print('Average train loss: {}'.format(trainloss))
        print('Average test loss: {}'.format(valloss))
        # dice = check_accuracy(test_loader, model, device=device)
        loss_train.append(trainloss)
        loss_val.append(valloss)
        print()
        
        scheduler.step()

        if valloss < minvalidation and args.save_model:
            minvalidation = valloss

            save_model(model, is_best=True, checkpoint_dir='checkpoints')

    # plot_train_loss(loss_train, loss_val)


if __name__ == '__main__':
    main()
