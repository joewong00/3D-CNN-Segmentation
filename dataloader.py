import h5py
from torch.utils.data import Dataset
from utils.transform_3d import Transform_3D_Mask_Label
from utils.utils import read_data_from_h5

class MRIDataset(Dataset):
    """Class for custom dataset loading"""

    def __init__(self, train=True, transform=None, elastic=True):
        super().__init__()

        # Specify path to the respective h5 files
        self.h5ftrain = './dataset/T2train.h5'
        self.h5ftrainmask = './dataset/T2trainmask.h5'
        self.h5ftest = './dataset/T2test.h5'
        self.h5ftestmask = './dataset/T2testmask.h5'

        self.train = train
        self.transform = transform
        self.elastic = elastic


    def __getitem__(self, index):
    
        normalize = None

        if self.train:
            x = read_data_from_h5(self.h5ftrain, index+1, tensor=False)
            y = read_data_from_h5(self.h5ftrainmask, index+1, tensor=False)

            # Comment this line if no normalization are needed
            normalize = (12870.1807,11750.7428)
            
        else:
            x = read_data_from_h5(self.h5ftest, index+1, tensor=False)
            y = read_data_from_h5(self.h5ftestmask, index+1, tensor=False)

            # Comment this line if no normalization are needed
            normalize = (12040.5588,10963.0117)

        # Custom transform for 3D data
        transform = Transform_3D_Mask_Label(self.transform, self.elastic, normalize=normalize)
        x,y = transform(x,y)
        
        return x, y


    def __len__(self):

        trainfile = h5py.File(self.h5ftrain,'r')
        testfile = h5py.File(self.h5ftest,'r')

        if self.train:
            return len(trainfile)
        else:
            return len(testfile)


