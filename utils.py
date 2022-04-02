from torch.utils.data import DataLoader, random_split
from dataloader import MRIDataset

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nibabel as nib
import re
import h5py

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    # Save checkpoint of model

    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    # Load checkpoint of model

    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(train=True, transform=True, **kwargs):
    # Get the train dataset or test dataset loader

    dataset = MRIDataset(train=train, transform=transform)

    # train/val split
    if train:
        # train_set, val_set = random_split(dataset, [int(len(dataset)*0.9),int(len(dataset)*0.1)])
        
        trainloader = DataLoader(dataset=dataset, **kwargs)
        # valloader = DataLoader(dataset=val_set, **kwargs)

        return trainloader

    # test
    else:
        testloader = DataLoader(dataset=dataset, **kwargs)
        return testloader


def dice_coefficient(prediction, truth):

   return np.sum(prediction[truth==1]) * 2.0 / (np.sum(prediction) + np.sum(truth))


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
    return dice_score/len(loader)


def plotloss(outfile):
    # Plotting loss (train and validation) based on the log file

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
    # Plotting accuracy (test accuracy and dice score) based on the log file

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
    # Compute performance average from the test set

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


def visualize2d(data, depth):
	# Visualize Single Image Data

    assert depth in data.shape, 'The input "depth" is not compatible with the data'
    assert data.shape == 4, 'Data must contain 4 dimensions: (CxDxWxH)'
    assert isinstance(data, np.ndarray) or torch.is_tensor(data), 'Data must be either numpy or torch tensors'

	# Convert to cpu
    data = data.cpu()

    for i in range(depth):
        plt.imshow(data[0,0,i,:,:],cmap='gray')
        plt.axis('off')

    plt.show()
    plt.savefig('data.png')


def add_mask_colour(mask, colour="red"):

    if torch.is_tensor(mask):
        mask = mask.numpy()

    # Make sure the input has shape (C * D * W * H)
    if len(mask.shape) == 5:
        mask = np.squeeze(mask, 0)

    kernel = mask.astype(bool)
    mask[kernel] = 255

    if colour == "red":
        mask = np.pad(mask, ((0,2),(0,0),(0,0),(0,0)))

    elif colour == "green":
        mask = np.pad(mask, ((1,1),(0,0),(0,0),(0,0)))

    elif colour == "blue":
        mask = np.pad(mask, ((2,0),(0,0),(0,0),(0,0)))

    else:
        raise Exception("Invalid colour, the colour must be either (red, green, blue).") 

    # Make shape D * W * H * C
    mask = np.moveaxis(mask, 0, 3)
    mask = mask.astype(np.uint8)

    return mask


def read_data_as_numpy(image_path):
    # Read data and convert to numpy array

    image_obj = nib.load(image_path)

    # Extract data as numpy array
    image_data = image_obj.get_fdata()

    return image_data


def read_data_from_h5(file_path, index, tensor=True):
    # Read data from h5 file and convert to numpy/torch tensor

    h5f = h5py.File(file_path,'r')
    data = h5f[f'T2data_{index}'][:]
    
    if tensor:
        data = torch.from_numpy(data)

    return data


def plot_sidebyside(feature, prediction, groundtruth, depth):
    # Plotting the original data, prediction and groundtruth in 3 columns (each row represents the depth)

    f, axarr = plt.subplots(14,3,figsize=(100,100))

    if torch.is_tensor(prediction):
        prediction = prediction.numpy()

    # Make sure the input has shape (C * D * W * H)
    if len(prediction.shape) > 3:
        prediction = np.squeeze(prediction)

    for i in range(depth):
        axarr[i,0].imshow(feature[0,0,i,:,:],cmap='gray')
        axarr[i,1].imshow(prediction[0,0,i,:,:],cmap='gray')
        axarr[i,2].imshow(groundtruth[0,0,i,:,:],cmap='gray')
        plt.axis('off')


def plot_train_loss(loss_train, loss_val):
    plt.plot(loss_train, label="Train loss")
    plt.plot(loss_val, label="Val loss")
    plt.legend()
    plt.show()
    plt.savefig('Training Loss.png')


def plot_train_accuracy(acc_train, acc_val):
    plt.plot(acc_train, label="Train loss")
    plt.plot(acc_val, label="Val loss")
    plt.legend()
    plt.show()
    plt.savefig('Training Loss.png')


def number_of_features_per_level(init_channel_number, num_levels):
    # Return a list of features
    return [init_channel_number * 2 ** k for k in range(num_levels)]