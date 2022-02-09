import matplotlib.pyplot as plt
from dataloader import MRIDataset
from residual3dunet.model import ResidualUNet3D
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
#import cv2
import os

def prepare_plot(origImage, origMask, predMask):
	# initialize our figure
	figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
	# plot the original image, its mask, and the predicted mask
	ax[0].imshow(origImage)
	ax[1].imshow(origMask)
	ax[2].imshow(predMask)
	# set the titles of the subplots
	ax[0].set_title("Image")
	ax[1].set_title("Original Mask")
	ax[2].set_title("Predicted Mask")
	# set the layout of the figure and display it
	figure.tight_layout()
	figure.show()

device = torch.device('cpu')
model = ResidualUNet3D(in_channels=1, out_channels=1, testing=True).to(device)
model.load_state_dict(torch.load("model.pt", map_location=device))
    
dataset2 = MRIDataset(train=False, transform=True)
test_loader = DataLoader(dataset = dataset2, batch_size=1, shuffle=True)

for data, target in test_loader:
    data, target = data.float().to(device), target.float().to(device)
    output = model(data)
    break

preds = (output > 0.5).float()

batch, channel, depth, width, height = preds.shape
print(preds.shape)
print("Prediction")
for i in range(depth):
    plt.imshow(preds[0,0,i,:,:],cmap='gray')
    plt.axis('off')
    plt.show()

print("Target")
for i in range(depth):
    plt.imshow(target[0,0,i,:,:],cmap='gray')
    plt.axis('off')
    plt.show()