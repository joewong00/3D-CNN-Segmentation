from calendar import day_abbr
import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from dataloader import MRIDataset


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(train=True, transform=True, **kwargs):

    dataset = MRIDataset(train=train, transform=transform)

    # train/val split
    if train:
        train_set, val_set = random_split(dataset, [int(len(dataset)*0.9),int(len(dataset)*0.1)])
        
        trainloader = DataLoader(dataset=train_set, **kwargs)
        valloader = DataLoader(dataset=val_set, **kwargs)

        return trainloader, valloader

    # test
    else:
        testloader = DataLoader(dataset=dataset, **kwargs)
        return testloader


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.float().to(device)
            y = y.float().to(device)
            preds = torch.sigmoid(model(x))

            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)

            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    #model.train()


def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):

    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.float().to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y, f"{folder}{idx}.png")

    model.train()