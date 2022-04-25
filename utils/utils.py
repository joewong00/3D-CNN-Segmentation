from matplotlib.backends.backend_pdf import PdfPages

import torch
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import matplotlib.patches as mpatches
import pandas as pd
import nibabel as nib
import re
import h5py
import os



########################################### PREPROCESS ###########################################

def preprocess(data, rotate=True, to_tensor=True, normalize=False):
    """Preprocess a single input data (to depth first, rotate, make depth 14, to tensor)
    Args: 
        data (np.ndarray): 3D or 4D numpy array (W * H * D) or (C * W * H * D)
        rotate (bool): Rotate the input image 90 degree (W,H) - > (H,W)
        to_tensor (bool): Convert preprocessed data to torch tensors
        normalize (bool): Normalize the data to have a mean of 0 and std of 1 (normalization is required when using R2U-Net)
    Returns:
        data (np.ndarray | torch.tensor): Processed Data
    """
    assert len(data.shape) in (3,4), 'Data must contain 3 or 4 dimensions: (W*H*D) or (C*W*H*D)'
    assert isinstance(data, np.ndarray), 'Data must be either numpy array'

    if len(data.shape) > 3:
        data = np.squeeze(data)

    data = to_depth_first(data)

    # rotate the data image 90 degree (W * H -> H * W)
    if rotate:
        data = np.moveaxis(data,1,2)
    
    # Make depth 14 (pad the image at the depth axis with 0)
    depth_pad = 14 - data.shape[0]
    data = np.pad(data, ((0,depth_pad),(0,0),(0,0)))

    # Normalize data
    if normalize:
        # Test normalization
        data = (data - data.mean())/data.std()

    # Convert to torch tensors
    if to_tensor:
        data = torch.from_numpy(data)

    return data


    
def write_to_h5(dir, out_filename):
    """Write all data from the directory to a h5 file
    Args:
        dir (pathlike): path to the dataset directory
        out_filename (string): output file name
    """

    # Create output h5 file
    hf = h5py.File('./dataset/'+out_filename+'.h5','w')

    # For every file in dataset directory
    for file in sorted(os.listdir(dir)):
        if file.endswith(".nii.gz"):

            filename = file.split(".")[0]
            data = read_data_as_numpy(os.path.join(dir,file))
            data = preprocess(data)
            data = add_channel(data)

            hf.create_dataset(f'{filename}', data=data)

    hf.close()



########################################### MODEL ###########################################

def save_model(state, is_best, checkpoint_dir):
    """Saves model state dicts at '{checkpoint_dir}/last_model.pt' or '{checkpoint_dir}/model.pt'.
    Args:
        state (torch.nn.Module): trained model
        is_best (bool): if True state contains the best model seen so far
        checkpoint_dir (string): directory where the checkpoint are to be saved
    """
    # If path does not exist then create directory
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    # Save the best model
    if is_best:
        best_file_path = os.path.join(checkpoint_dir, 'model.pt')

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


def predict(model,input,threshold,device):

	model.eval()

	input = to_depth_first(input)

	if len(input.shape) == 3:
		input = add_channel(input)

	# Add batch dimension
	input = input.unsqueeze(0)
	input = input.to(device=device, dtype=torch.float32)

	# Disable grad
	with torch.no_grad():

		output = model(input)
		preds = (output > threshold).float()

		# Squeeze channel and batch dimension
		preds = torch.squeeze(preds)

		# Convert to numpy
		preds = preds.cpu().numpy()

	return preds



########################################### EVALUATION ###########################################

def plot_train_loss(loss_train, loss_val, title="Training and Validation Loss", x_label="Epoch", y_label="Loss"):
    """Plot the graph of loss during training, x value = number of epoch.
    Args: 
        loss_train (list): list of loss value for training set for every epoch
        loss_val (list): list of loss value for validation set for every epoch
    """
    plt.plot(loss_train, label="Train loss")
    plt.plot(loss_val, label="Val loss")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()
    plt.savefig('Training Loss.png')



def plot_train_accuracy(acc_train, acc_val, title="Training and Validation Accuracy", x_label="Accuracy", y_label="Epoch"):
    """Plot the graph of accuracy during training, x value = number of epoch.
    Args: 
        acc_train (list): list of accuracy for training set for every epoch
        acc_val (list): list of accuracy value for validation set for every epoch
    """
    plt.plot(acc_train, label="Train accuracy")
    plt.plot(acc_val, label="Val accuracy")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()
    plt.savefig('Training Accuracy.png')



def plot_loss_from_log(log, title="Train 1"):
    """Plot the graph of loss from training output log file, x value = number of epoch.
    Args: 
        log (path like): path to the training log file
    """

    # Using regular expression to extract the loss value for each epoch
    with open(log) as f:
        fin = f.read()
        testloss = re.findall(r"(test\sloss: )(\d.\d+e?-?\d+)", fin)
        trainloss = re.findall(r"(train\sloss: )(\d.\d+e?-?\d+)", fin)

    loss_test = []
    loss_train = []

    for i in range(len(testloss)):
        loss_test.append(float(testloss[i][1]))
        loss_train.append(float(trainloss[i][1]))

    plot_train_loss(loss_train, loss_test, title=title)



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
    std = []

    # Evaluation metrics
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
        statlist = list(stat[key] for stat in dicts[startidx:endidx])
        std.append(np.std(statlist))

        stats[key] = total/length

    # convert into dataframe
    if dataframe:
        stats = pd.DataFrame(stats.items(), columns=['Metric','Score'])
        stats['Standard Deviation'] = std

    return stats


def bland_altman_plot(data1, data2, xlabel='Means', ylabel='Difference', title='Bland-Altman Plot', savefig=False, filename='bland-altman'):
    """Bland Altman plot based on the 2 data
    
    Args:
        data1 (np.adarray): data 1 in the form of 1-dimensional numpy array 
        data2 (np.adarray): data 2 in the form of 1-dimensional numpy array
        xlabel (str): x-axis label
        ylabel (str): y-axis label
        title (str): plot title
        savefig (bool): If True, save the plot with the filename specified
        filename (str): Output file name for the plot
    """

    f,ax = plt.subplots(1,figsize=(8,5))

    sm.graphics.mean_diff_plot(data1,data2,ax=ax,limit_lines_kwds={'color':'red'})

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.subplots_adjust(top=0.8)

    if savefig:
        if not os.path.exists('output'):
            os.mkdir('output')
        plt.savefig(os.path.join('output',filename+'.png'))

    plt.show()
    


########################################### VISUALIZATION ###########################################

def visualize2d(data, size=(3,3)):
    """Visualize 3d data and save images to image.pdf。
    Args:
        data (np.ndarray): image data in the shape of (W * H * D) or (C * W * H * D)
        size (tuple): image size (W * H)
    """
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
    pdf = PdfPages("output/image.pdf")

    # Plot 3D data depth-wise
    for i in range(depth):
        fig = plt.figure(figsize=size)
        plt.title('Data')
        plt.imshow(data[:,:,i],cmap='gray')
        plt.axis('off')
        plt.show()

        pdf.savefig(fig)

    pdf.close()



def add_mask_colour(mask, colour="red"):
    """Applying colour to the mask (red, blue or green).
    Args:
        mask (np.ndarray/torch.tensors): A 3D (W * H * D) or a 4D (C * W * H * D) array/tensor
        colour (string): red, green, or blue
    Returns:
        mask (np.ndarray): A 4D numpy array with channel last (W * H * D * C)
    """

    # Shape: (C*W*H*D)
    if len(mask.shape) == 3:
        mask = add_channel(mask)

    mask = convert_to_numpy(mask)

    # Pad RGB channel accordingly
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
    mask = mask.astype(np.float64)

    return mask



def greytoRGB(data):
    """Convert image data from greyscale to RGB images (add RGB channel)
    Args:
        data (np.ndarray/torch.tensors): a 3D array/tensor image data, containing single or zero channel
    Returns:
        RGB_data (np.ndarray): a 4D array (D * W * H * 3), added 3 channel dimensions at the end of the array
    """
    data = convert_to_numpy(data)

    # Remove channel
    if len(data.shape) > 3:
        data = np.squeeze(data)

    # normalize data
    data = (data - np.min(data)) / (np.max(data) - np.min(data))

    data = np.stack((data,data,data), axis=3)
    data = data.astype(np.float64)

    return data



def plot_sidebyside(feature, prediction, target, save_file=True):
    """Plot the feature, predicted mask and groundtruth side by side to compare.
    Args:
        feature (np.ndarray/torch.tensors): A 3D or a 4D array/tensor, the original feature image
        prediction (np.ndarray/torch.tensors): A 3D or a 4D array/tensor, the predicted mask
        target (np.ndarray/torch.tensors): A 3D or a 4D array/tensor, the ground truth mask
    """

    assert len(feature.shape) in (3,4), 'feature must contain 3 or 4 dimensions: (W*H*D) or (C*W*H*D)'
    assert len(prediction.shape) in (3,4), 'prediction must contain 3 or 4 dimensions: (W*H*D) or (C*W*H*D)'
    assert len(target.shape) in (3,4), 'target must contain 3 or 4 dimensions: (W*H*D) or (C*W*H*D)'

    f, axarr = plt.subplots(14,3,figsize=(50,50))

    f.suptitle('Output Comparison (Feature | Prediction | GroundTruth)', fontsize=50)

    # Preprocessing feature, prediction and target
    feature = np.squeeze(convert_to_numpy(feature))
    prediction = np.squeeze(convert_to_numpy(prediction))
    target = np.squeeze(convert_to_numpy(target))

    feature = to_depth_last(feature)
    prediction = to_depth_last(prediction)
    target = to_depth_last(target)
    
    depth = prediction.shape[-1]

    for i in range(depth):
        axarr[i,0].imshow(feature[:,:,i],cmap='gray')
        axarr[i,1].imshow(prediction[:,:,i],cmap='gray')
        axarr[i,2].imshow(target[:,:,i],cmap='gray')
        

    if save_file:
        # Save image to pdf
        pdf = PdfPages("output/compare.pdf")
        pdf.savefig(f)
        pdf.close()



def plot_overlapped(feature, prediction, target, output_dir='output'):
    """Plot the feature, predicted mask and groundtruth overlapping each other.
    Args:
        feature (np.ndarray/torch.tensors): A 3D or a 4D array/tensor, the original feature image
        prediction (np.ndarray/torch.tensors): A 3D or a 4D array/tensor, the predicted mask
        target (np.ndarray/torch.tensors): A 3D or a 4D array/tensor, the ground truth mask
        output_dir (path like): Output plot directory
    """

    assert len(feature.shape) in (3,4), 'feature must contain 3 or 4 dimensions: (W*H*D) or (C*W*H*D)'
    assert len(prediction.shape) in (3,4), 'prediction must contain 3 or 4 dimensions: (W*H*D) or (C*W*H*D)'
    assert len(target.shape) in (3,4), 'target must contain 3 or 4 dimensions: (W*H*D) or (C*W*H*D)'

    # Preprocessing feature, prediction and target
    feature = np.squeeze(convert_to_numpy(feature))
    prediction = np.squeeze(convert_to_numpy(prediction))
    target = np.squeeze(convert_to_numpy(target))

    feature = greytoRGB(to_depth_last(feature))
    prediction = to_depth_last(prediction)
    target = to_depth_last(target)

    depth = prediction.shape[-1]

    # Add colour to masks    
    prediction = add_mask_colour(prediction, "red")
    target = add_mask_colour(target, "blue")

    overlap = prediction + target

    # Set masked pixels as 1
    feature[overlap.astype(bool)] = 1

    # Colour labelling
    colors = [(1,0,0,1), (0,0,1,1), (1,0,1,1)]
    values = ['prediction', 'target', 'overlapped']
    patches = [ mpatches.Patch(color=colors[i], label=values[i] ) for i in range(len(values)) ]

    # Save image to pdf
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    pdf = PdfPages(os.path.join(output_dir, 'result.pdf'))

    # Plot 3D data depth-wise
    for i in range(depth):
        fig = plt.figure()
        plt.title('Output')
        plt.imshow(feature[:,:,i,:])
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
        plt.axis('off')
        plt.show()

        pdf.savefig(fig)

    pdf.close()




########################################### MISC ###########################################


def channel_exist(data):
    # Check if channel dimension exist in the data
    return 1 in data.shape
    

def add_channel(data, dim=0):
    # Add channel dimension to the first dimension of 3D data

    assert len(data.shape) == 3, 'Data must contain only 3 dimensions (W, H, D)'
    assert isinstance(data, np.ndarray) or torch.is_tensor(data), 'Data must be either numpy array or torch tensors'

    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)

    data = torch.unsqueeze(data, dim)

    return data


def to_depth_first(data):
    # Reshape the data image from depth last to depth first

    data = convert_to_numpy(data)
    data = np.squeeze(data)

    if data.shape[-1] < data.shape[-2]:
        data = np.moveaxis(data,-1,0)

    return data
    

def to_depth_last(data):
    # Reshape the data image from depth first to depth last

    data = convert_to_numpy(data)
    data = np.squeeze(data)

    if data.shape[0] < data.shape[1]:
        data = np.moveaxis(data,0,-1)

    return data


def convert_to_numpy(data):
    # Convert torch tensor to numpy array

    assert isinstance(data, np.ndarray) or torch.is_tensor(data), 'Data must be either numpy array or torch tensors'

    if torch.is_tensor(data):
        data = data.numpy()

    return data


def read_data_as_numpy(image_path):
    # Read data and convert to numpy array (data in the format of nii.gz)

    assert os.path.exists(image_path), 'Path not found'

    # Extract data as numpy array
    image_obj = nib.load(image_path)
    image_data = image_obj.get_fdata()

    return image_data


def read_data_from_h5(file_path, index, tensor=True):
    # Read data from h5 file and convert to numpy/torch tensor, make sure "mask" presents in file_path to read mask

    assert index in range(1,51), 'Only maximum of 50 data available'

    h5f = h5py.File(file_path,'r')

    if index < 10:
        index = '0'+str(index)

    if 'mask' in file_path.casefold():
        data = h5f[f'MRI'+str(index)+'_T2mask'][:]
    else:
        data = h5f[f'MRI'+str(index)+'_T2'][:]
    
    if tensor:
        data = torch.from_numpy(data)

    return data


def numpy_to_nii(data):
    # Convert numpy array and set it to nii format

    img = nib.Nifti1Image(data, np.eye(4))

    return img

