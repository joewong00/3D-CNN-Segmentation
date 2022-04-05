from torch.utils.data import DataLoader, random_split
from dataloader import MRIDataset
from matplotlib.backends.backend_pdf import PdfPages

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
import pandas as pd
import nibabel as nib
import re
import h5py
import shutil
import os

########################################### MODEL ###########################################

def save_model(state, is_best, checkpoint_dir):
    """Saves model state dicts at '{checkpoint_dir}/last_model.pt' or '{checkpoint_dir}/model.pt'.
    Args:
        state (torch.nn.Module): trained model
        is_best (bool): if True state contains the best model seen so far
        checkpoint_dir (string): directory where the checkpoint are to be saved
    """
    # If path does o
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    # Save the best model
    if is_best:
        best_file_path = os.path.join(checkpoint_dir, 'model.pt')

        if os.path.exists(best_file_path):  # checking if there is a file with this name
            os.remove(best_file_path)  # deleting the file
            torch.save(state.state_dict(), best_file_path)
    
    else:
        last_file_path = os.path.join(checkpoint_dir, 'last_model.pt')
        torch.save(state.state_dict(), last_file_path)


def load_checkpoint(checkpoint_path, model, device):
    """Loads model and training parameters from a given checkpoint_path
    If optimizer is provided, loads optimizer's state_dict of as well.
    Args:
        checkpoint_path (string): path to the checkpoint to be loaded
        model (torch.nn.Module): model into which the parameters are to be copied
    Returns:
        state
    """
    if not os.path.exists(checkpoint_path):
        raise IOError(f"Checkpoint '{checkpoint_path}' does not exist")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))


def get_loaders(train=True, transform=None, elastic_transform=True, split=False, **kwargs):
    """Obtain the dataset loader (train or test set).
    Args:
        train (bool): if True return training dataset
        transform (torchvision.transforms): apply transformation to data
        elastic_transform (bool): if True apply elastic transformation to data
        split (bool): if True perform train-val set splitting randomly 90/10
        **kwargs : other dataloader parameters
    Returns:
        dataset loader (train, val) | test
    
    """
    
    # Get the train dataset or test dataset loader
    dataset = MRIDataset(train=train, transform=transform, elastic=elastic_transform)

    # train/val split
    if split:
        train_set, val_set = random_split(dataset, [int(len(dataset)*0.9),int(len(dataset)*0.1)])
        trainloader = DataLoader(dataset=train_set, **kwargs)
        valloader = DataLoader(dataset=val_set, **kwargs)

        return trainloader, valloader
    
    # train set / test set
    else:
        dataloader = DataLoader(dataset=dataset, **kwargs)

        return dataloader


def number_of_features_per_level(init_channel_number, num_levels):
    """Return a list of features, doubling in size depending on the num_levels.
    Args:
        init_channel_number (int): initial channel number
        num_levels (int): number of levels of the deep network
    Returns:
        list of features (lists)
    eg. number_of_features_per_level(64,4) -> [64,128,256,512]
    """
    return [init_channel_number * 2 ** k for k in range(num_levels)]



########################################### EVALUATION ###########################################

# def dice_coefficient(pred, target):

#     smooth = 1.

#     iflat = pred.contiguous().view(-1)
#     tflat = target.contiguous().view(-1)
#     intersection = (iflat * tflat).sum()

#     A_sum = torch.sum(tflat * iflat)
#     B_sum = torch.sum(tflat * tflat)
    
#     return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )


# def check_accuracy(loader, model, device="cuda"):
#     num_correct = 0
#     num_pixels = 0
#     dice_score = 0
#     model.eval()

#     with torch.no_grad():
#         for x, y in loader:
#             x = x.float().to(device)
#             y = y.float().to(device)
#             preds = torch.sigmoid(model(x))

#             preds = (preds > 0.5).float()

#             num_correct += (preds == y).sum()
#             num_pixels += torch.numel(preds)

#             dice_score += dice_coefficient(preds, y)
            
#     print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
#     print(f"Dice score: {dice_score/len(loader)}")
#     return dice_score/len(loader)


def plot_train_loss(loss_train, loss_val):
    """Plot the graph of loss during training, x value = number of epoch.
    Args: 
        loss_train (list): list of loss value for training set for every epoch
        loss_val (list): list of loss value for validation set for every epoch
    """
    plt.plot(loss_train, label="Train loss")
    plt.plot(loss_val, label="Val loss")
    plt.legend()
    plt.show()
    plt.savefig('Training Loss.png')


def plot_train_accuracy(acc_train, acc_val):
    """Plot the graph of accuracy during training, x value = number of epoch.
    Args: 
        acc_train (list): list of accuracy for training set for every epoch
        acc_val (list): list of accuracy value for validation set for every epoch
    """
    plt.plot(acc_train, label="Train accuracy")
    plt.plot(acc_val, label="Val accuracy")
    plt.legend()
    plt.show()
    plt.savefig('Training Accuracy.png')


def plot_loss_from_log(log):
    """Plot the graph of loss from training output log file, x value = number of epoch.
    Args: 
        log (path like): path to the training log file
    """

    # Using regular expression to extract the loss value for each epoch
    with open(log) as f:
        fin = f.read()
        testloss = re.findall(r"(test\sloss: )(\d.\d+)", fin)
        trainloss = re.findall(r"(train\sloss: )(\d.\d+)", fin)

    loss_test = []
    loss_train = []

    for i in range(len(testloss)):
        loss_test.append(float(testloss[i][1]))
        loss_train.append(float(trainloss[i][1]))

    plot_train_loss(loss_train, loss_test)


def plot_accuracy_from_log(log):
    """Plot the graph of accuracy and dice score from training output log file, x value = number of epoch.
    Args: 
        log (path like): path to the training log file
    """

    # Plotting accuracy (test accuracy and dice score) based on the log file
    with open(log) as f:
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
    """Compute average performance values from the test dataset. 
    Args:
        dicts (dict): list of evaluation metrics in the form of python dictionary
        startidx (int): start index of the list of dicts
        endidx (int): end index of the list of dicts
        dataframe (bool): if True then print the evaluation metrics in the form of dataframe
    Returns:
        stats (dict/dataframe): the average evaluation metrics
    """

    assert endidx != 0, 'Index cannot end at 0'

    stats = {}

    # Evaluation 
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


########################################### VISUALIZATION ###########################################

def visualize2d(data, size=(3,3)):
	# Visualize Single Image Data

    assert len(data.shape) in (3,4), 'Data must contain 3 or 4 dimensions: (W*H*D) or (C*W*H*D)'
    assert isinstance(data, np.ndarray) or torch.is_tensor(data), 'Data must be either numpy array or torch tensors'

    data = convert_to_numpy(data)
    data = to_depth_last(data)

    depth = data.shape[-1]

    # Convert to 3D
    if len(data.shape) == 4:
        data = np.squeeze(data)

    # Save image to pdf
    pdf = PdfPages("image.pdf")

    # Plot 3D data depth-wise
    for i in range(depth):
        fig = plt.figure(figsize=size)
        plt.imshow(data[:,:,i],cmap='gray')
        plt.axis('off')
        plt.show()

        pdf.savefig(fig)

    pdf.close()



def add_mask_colour(mask, colour="red"):

    # Shape: (C*W*H*D)
    if len(mask.shape) == 3:
        mask = add_channel(mask)

    mask = convert_to_numpy(mask)

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

    # Make shape W * H * D * C
    mask = np.moveaxis(mask, 0, 3)
    mask = mask.astype(np.uint8)

    return mask


def plot_result(feature, prediction, target, depth):
    # Plotting the original data, prediction and target in 3 columns (each row represents the depth)

    assert len(feature.shape) == 4, "Feature, prediction and target must have 4 dimensions (C,D,W,H)"
    assert feature.shape == prediction.shape == target.shape, "Feature, prediction and target must have the same dimensions"

    f, axarr = plt.subplots(14,3,figsize=(100,100))

    if torch.is_tensor(prediction):
        prediction = prediction.numpy()

    for i in range(depth):
        axarr[i,0].imshow(feature[0,i,:,:],cmap='gray')
        axarr[i,1].imshow(prediction[0,i,:,:],cmap='gray')
        axarr[i,2].imshow(target[0,i,:,:],cmap='gray')
        plt.axis('off')


def plot_overlapped(feature, prediction, target, size=(3,3)):
    # Plotting the data, prediction and target, overlapping each other

    assert len(feature.shape) in (3,4), 'feature must contain 3 or 4 dimensions: (W*H*D) or (C*W*H*D)'
    assert len(prediction.shape) in (3,4), 'prediction must contain 3 or 4 dimensions: (W*H*D) or (C*W*H*D)'
    assert len(target.shape) in (3,4), 'target must contain 3 or 4 dimensions: (W*H*D) or (C*W*H*D)'

    feature = convert_to_numpy(feature)
    feature = to_depth_last(feature)

    prediction = convert_to_numpy(prediction)
    prediction = to_depth_last(prediction)

    target = convert_to_numpy(target)
    target = to_depth_last(target)

    depth = feature.shape[-1]

    # Convert to 3D
    feature = np.squeeze(feature)
    prediction = np.squeeze(prediction)
    target = np.squeeze(target)

    prediction = add_mask_colour(prediction, "red")
    target = add_mask_colour(target, "blue")

    overlap = prediction + target

    # Save image to pdf
    pdf = PdfPages("result.pdf")

    # Plot 3D data depth-wise
    for i in range(depth):
        fig = plt.figure(figsize=size)
        plt.imshow(feature[:,:,i],cmap='gray')
        plt.imshow(overlap[:,:,i,:],alpha=0.4)
        plt.axis('off')
        plt.show()

        pdf.savefig(fig)

    pdf.close()




########################################### MISC ###########################################


def channel_exist(data):
    # Check if channel dimension exist in the data
    return 1 in data.shape
    
def add_channel(data):
    # Add channel dimension to the first dimension of 3D data

    assert len(data.shape) == 3, 'Data must contain only 3 dimensions (W, H, D)'
    assert isinstance(data, np.ndarray) or torch.is_tensor(data), 'Data must be either numpy array or torch tensors'

    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)

    data = torch.unsqueeze(data, 0)

    return data


def to_depth_first(data):
    # Reshape the data image from depth last to depth first

    data = convert_to_numpy(data)

    if len(data.shape) == 3:
        data = np.moveaxis(data,-1,0)

    elif len(data.shape) == 4:
        data = np.moveaxis(data,-1,1)

    return data
    

def to_depth_last(data):
    # Reshape the data image from depth first to depth last

    data = convert_to_numpy(data)

    if len(data.shape) == 3:
        if data.shape[0] < data.shape[1]:
            data = np.moveaxis(data,0,-1)

    elif len(data.shape) == 4:
        if data.shape[1] < data.shape[2]:
            data = np.moveaxis(data,1,-1)

    return data


def convert_to_numpy(data):
    # Convert torch tensor to numpy array

    assert isinstance(data, np.ndarray) or torch.is_tensor(data), 'Data must be either numpy array or torch tensors'

    if torch.is_tensor(data):
        data = data.numpy()

    return data


def read_data_as_numpy(image_path):
    # Read data and convert to numpy array

    image_obj = nib.load(image_path)

    # Extract data as numpy array
    image_data = image_obj.get_fdata()

    # Exchange width and height dimension
    image_data = np.moveaxis(image_data,0,1)

    return image_data


def read_data_from_h5(file_path, index, tensor=True):
    # Read data from h5 file and convert to numpy/torch tensor

    h5f = h5py.File(file_path,'r')
    data = h5f[f'T2data_{index}'][:]
    
    if tensor:
        data = torch.from_numpy(data)

    return data


def numpy_to_nii(data):
    # Convert numpy array and set it to nii format

    img = nib.Nifti1Image(data, np.eye(4))

    return img

