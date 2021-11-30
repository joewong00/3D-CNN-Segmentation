import h5py
import torch
from torch.utils.data import Dataset


class MRIDataset(Dataset):
    def __init__(self, train=True, transform=True):
        super().__init__()

        self.h5ftrain = h5py.File('T2train.h5','r')
        self.h5ftrainmask = h5py.File('T2trainmask.h5','r')
        self.h5ftest = h5py.File('T2test.h5','r')
        self.h5ftestmask = h5py.File('T2testmask.h5','r')

        self.train = train
        self.transform = transform

    def __getitem__(self, index):
    
        if self.train:
            X = self.h5ftrain[f'T2data_{index+1}'][:]
            Y = self.h5ftrainmask[f'T2maskdata_{index+1}'][:]
        else:
            X = self.h5ftest[f'T2data_{index+1}'][:]
            Y = self.h5ftestmask[f'T2maskdata_{index+1}'][:]

        if self.transform:
            X = torch.from_numpy(X)
            Y = torch.from_numpy(Y)

        return X,Y 

    def __len__(self):

        if self.train:
            return len(self.h5ftrain)
        else:
            return len(self.h5ftest)