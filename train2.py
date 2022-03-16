from __future__ import print_function
from webbrowser import get
from torch.optim import Adam
from dataloader import MRIDataset
from residual3dunet.model import ResidualUNet3D
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
from torch.nn import DataParallel
from utils import get_loaders, check_accuracy, load_checkpoint, save_checkpoint, save_predictions_as_imgs 

import torch
import argparse
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T


# Training
def train(args, model, device, train_loader, optimizer, epoch):
    costs = []
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.float().to(device), target.float().to(device)
        optimizer.zero_grad()
        output = model(data)
        cost = F.binary_cross_entropy_with_logits(output, target)
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

def test(model, device, test_loader, epoch):
    costs = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.float().to(device), target.float().to(device)
            output = model(data)
            cost = F.binary_cross_entropy_with_logits(output, target)

            costs.append(cost.item())

            print('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(test_loader.dataset),
                100. * batch_idx / len(test_loader), cost.item()))

        avgcost = sum(costs)/len(costs)
    
    return avgcost



# def test(model, device, test_loader):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.float().to(device), target.float().to(device)
#             output = model(data)
#             test_loss += F.binary_cross_entropy_with_logits(output, target, reduction='sum').item()  # sum up batch loss
#             pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#             correct += pred.eq(target.view_as(pred)).sum().item()

#     test_loss /= len(test_loader.dataset)

#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch 3D Segmentation')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',help='input batch size for testing (default: 1000)')

    parser.add_argument('--epochs', type=int, default=3, metavar='N',
                        help='number of epochs to train (default: 14)')

    parser.add_argument('--lr', type=float, default=2.5e-4, metavar='LR',
                        help='learning rate (default: 1.0)')

    parser.add_argument('--gamma', type=float, default=0.1, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--multi-gpu', action='store_true', default=False,
                        help='use multiple gpu for training')

    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)


    # Data Loading
    traindataset = MRIDataset(train=True, transform=T.Compose([T.ToTensor(),
                                            T.RandomHorizontalFlip(),
                                            T.RandomCrop((240,240), padding=50, pad_if_needed=True),
                                           ]), elastic=True)

    testdataset = MRIDataset(train=False, transform=T.ToTensor())
                                        
    train_set, val_set = random_split(traindataset, [int(len(traindataset)*0.9),int(len(traindataset)*0.1)])

    train_loader = DataLoader(dataset=train_set, **train_kwargs)
    val_loader = DataLoader(dataset=val_set, **train_kwargs)
    test_loader = DataLoader(dataset=testdataset, **test_kwargs)

    # Model
    model = ResidualUNet3D(in_channels=1, out_channels=1).to(device)

    # If using multiple gpu
    if args.multi_gpu:
        model = DataParallel(model)

    # Hyperparameters
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=args.gamma)

    # Validation Loss
    minvalidation = 10

    # Training process
    for epoch in range(1, args.epochs + 1):
        trainloss = train(args, model, device, train_loader, optimizer, epoch)
        valloss = test(model, device, val_loader, epoch)

        print('Average train loss: {}'.format(trainloss))
        print('Average test loss: {}'.format(valloss))
        check_accuracy(test_loader, model, device=device)
        print()
        
        scheduler.step()

        if valloss < minvalidation and args.save_model:
            minvalidation = valloss
            torch.save(model.state_dict(), "model{}.pt".format(epoch))


if __name__ == '__main__':
    main()