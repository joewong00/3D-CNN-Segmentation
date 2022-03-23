import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from dataloader import MRIDataset
import matplotlib.pyplot as plt
import pandas as pd
import re


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


def dice_coefficient(pred, target):

    dice_score = 0
    dice_score += (2 * (pred * target).sum()) / (
                (pred + target).sum() + 1e-8
            )

    return dice_score


def iou(pred, target):
    
    pred = pred.int()
    target = target.int()

    intersection = (pred & target).float().sum()
    union = (pred | target).float().sum()   

    iou = (intersection + 1e-6) / (union + 1e-6) 

    return iou.mean()
    


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

            dice_score += dice_coefficient(preds, y)
            
    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
    print(f"Dice score: {dice_score/len(loader)}")


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


def plotloss(outfile):
    # Plotting loss (train and validation)

    with open(outfile) as f:
        fin = f.read()
        testloss = re.findall(r"(test\sloss: )(\d.\d+)", fin)
        trainloss = re.findall(r"(train\sloss: )(\d.\d+)", fin)

    loss_test = []
    loss_train = []

    for i in range(len(testloss)):
        loss_test.append(float(testloss[i][1]))
        loss_train.append(float(trainloss[i][1]))

    plt.plot(loss_train, label="Training loss")
    plt.plot(loss_test, label="Val loss")
    plt.legend()

def plotaccuracy(outfile):
    # Plotting accuracy (test accuracy and dice score)

    with open(outfile) as f:
        fin = f.read()
        accuracy = re.findall(r"(acc )(\d+.\d+)", fin)
        dice = re.findall(r"(score: )(\d.\d+)", fin)

    test_accuracy = []
    test_dice = []

    for i in range(len(accuracy)):
        test_accuracy.append(float(accuracy[i][1]))
        test_dice.append(float(dice[i][1]))

    test_accuracy = [i / 100 for i in test_accuracy]

    plt.plot(test_dice, label="Dice Score")
    plt.plot(test_accuracy, label="Accuracy")
    plt.legend()


def compute_average(dicts, startidx=None, endidx=None, dataframe=False):
    # Compute the average 

    assert endidx != 0, 'Index cannot end at 0'

    stats = {}

    metrics = ['Dice',
        'Jaccard',
        'Sensitivity',
        'Specificity',
        'Precision', 
        'Accuracy', 
        'Mean_Surface_Distance', 
        'Hausdorff_Distance', 
        'Volume_Difference']

    for key in metrics:
        total = sum(stat[key] for stat in dicts[startidx:endidx])
        length = len(dicts[startidx:endidx])
        stats[key] = total/length

    # convert into dataframe
    if dataframe:
        stats = pd.DataFrame(stats.items(), columns=['Metric','Score'])

    return stats