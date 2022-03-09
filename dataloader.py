import h5py
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as F
import random
import numpy as np
from elastic_transform import RandomElastic

class MRIDataset(Dataset):
    def __init__(self, train=True, transform=None):
        super().__init__()

        self.h5ftrain = h5py.File('dataset/T2train.h5','r')
        self.h5ftrainmask = h5py.File('dataset/T2trainmask.h5','r')
        self.h5ftest = h5py.File('dataset/T2test.h5','r')
        self.h5ftestmask = h5py.File('dataset/T2testmask.h5','r')

        self.train = train
        self.transform = transform


    def __getitem__(self, index):
    
        if self.train:
            X = self.h5ftrain[f'T2data_{index+1}'][:]
            Y = self.h5ftrainmask[f'T2maskdata_{index+1}'][:]
        else:
            X = self.h5ftest[f'T2data_{index+1}'][:]
            Y = self.h5ftestmask[f'T2maskdata_{index+1}'][:]


        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.transform is not None:
            X = np.squeeze(X)
            X = np.moveaxis(X, 0, 2)
            X = self.transform(X)
            X = torch.unsqueeze(X, 0)
            
        random.seed(seed) # apply this seed to target tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.transform is not None:
            Y = np.squeeze(Y)
            Y = np.moveaxis(Y, 0, 2)
            Y = self.transform(Y)
            Y = torch.unsqueeze(Y, 0)
        
        return X, Y


    def __len__(self):

        if self.train:
            return len(self.h5ftrain)
        else:
            return len(self.h5ftest)