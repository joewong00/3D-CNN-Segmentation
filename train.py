from re import I
from torch.optim import Adam
from dataloader import MRIDataset
from model.resunet3d import ResUNet3D
from model.r2unet3d import R2UNet3D
from model.unet3d import UNet3D
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
from torch.nn import DataParallel
from utils.utils import load_checkpoint, plot_train_loss, save_model
from utils.evaluate import evaluate
from utils.lossfunction import DiceBCELoss, DiceLoss, IoULoss, FocalLoss, FocalTverskyLoss, TverskyLoss, CoshLogDiceLoss

import torch
import argparse
import torchvision.transforms as T
import logging


# Training
def train(args, model, device, train_loader, optimizer, epoch, criterion):
    costs = []
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

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


# def test(model, device, test_loader, epoch, loss):
#     costs = []
#     model.eval()
#     with torch.no_grad():
#         for batch_idx, (data, target) in enumerate(test_loader):
#             data, target = data.float().to(device), target.float().to(device)
#             output = model(data)

#             cost = loss(output, target)

#             costs.append(cost.item())

#             print('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(test_loader.dataset),
#                 100. * batch_idx / len(test_loader), cost.item()))

#         avgcost = sum(costs)/len(costs)
    
#     return avgcost


def get_args():
    # Train settings
    parser = argparse.ArgumentParser(description='PyTorch 3D Segmentation')
    parser.add_argument('--network', '-u', default='Unet3D', help='Specify the network (Unet3D / ResidualUnet3D / R2Unet3D)')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',help='input batch size for training (default: 1)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=2.5e-4, metavar='LR', help='learning rate (default: 2.5e-4)')
    parser.add_argument('--gamma', type=float, default=0.1, metavar='M',help='Learning rate step gamma (default: 0.1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,help='For Saving the current Model')
    parser.add_argument('--checkpoint', '-c', metavar='FILE', help='Specify the path to the model')

    return parser.parse_args()


def main():

    # ------------------------------------ Network Config ------------------------------------

    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size, 'shuffle': True}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)


    assert args.network.casefold() in ("unet3d", "residualunet3d","r2unet3d"), 'Network must be either (Unet3D / ResidualUnet3D)'

	# Specify network
    if args.network.casefold() == "unet3d":
        model = UNet3D(in_channels=1, out_channels=1).to(device)
    elif args.network.casefold() == "residualunet3d":
        model = ResUNet3D(in_channels=1, out_channels=1).to(device)
    else:
        model = R2UNet3D(in_channels=1, out_channels=1).to(device)

    # If using multiple gpu
    if torch.cuda.device_count() > 1 and use_cuda:
        model = DataParallel(model)

    # If load checkpoint
    if args.checkpoint:
        load_checkpoint(args.checkpoint, model, device=device)

    # Hyperparameters
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=50, gamma=args.gamma)
    loss = DiceLoss()


    # ------------------------------------ Data Loading ------------------------------------

    # Train data transformation
    transformation = T.Compose([T.ToTensor(),
                    T.RandomHorizontalFlip()
                    # T.RandomCrop((240,240), padding=50, pad_if_needed=True)
                    ])

    traindataset = MRIDataset(train=True, transform=transformation, elastic=True)

    # Train validation set splitting 90/10
    # train_set, val_set = random_split(traindataset, [int(len(traindataset)*0.9),int(len(traindataset)*0.1)])

    test_kwargs = {'batch_size': args.batch_size}

    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': False}
        test_kwargs.update(cuda_kwargs)

    testdataset = MRIDataset(train=False, transform=T.ToTensor(), elastic=False)

	# Data Loading
    test_loader = DataLoader(dataset=testdataset, **test_kwargs)
    train_loader = DataLoader(dataset=traindataset, **train_kwargs)
 
    # ------------------------------------ Training Loop ------------------------------------

    # Validation Loss
    minvalidation = 1
    loss_train = []
    loss_val = []

    logging.info(f'''Starting training:
        Network:         {args.network}
        Epochs:          {args.epochs}
        Batch size:      {args.batch_size}
        Learning rate:   {args.lr}
        Training size:   {len(train_loader)}
        Validation size: {len(test_loader)}
        Device:          {device.type}
    ''')

    # Training process
    for epoch in range(1, args.epochs + 1):

        trainloss = train(args, model, device, train_loader, optimizer, epoch, loss)
        # valloss = test(model, device, val_loader, epoch, loss)

        valstat = evaluate(model, test_loader, device, 0.5, show_stat=False)
        valloss = 1 - valstat['Dice']

        print('Average train loss: {}'.format(trainloss))
        print('Average test loss: {}'.format(valloss))
        loss_train.append(trainloss)
        loss_val.append(valloss)
        print()
        
        scheduler.step()

        # Save the best validated model
        if valloss < minvalidation and args.save_model:
            minvalidation = valloss

            save_model(model, is_best=True, checkpoint_dir='checkpoints')

    # Plot training loss graph
    plot_train_loss(loss_train, loss_val)


if __name__ == '__main__':
    main()
