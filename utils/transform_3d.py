import numpy as np
import torch
import random
import torchvision.transforms as T

from elastic_transform import RandomElastic


class Transform_3D_Mask_Label(object):
    def __init__(self, transforms, elastic = True, normalize = None):

        assert isinstance(normalize, (type(None), tuple)), 'normalize must either be NoneType or Tuple of mean and std'

        self.transforms = transforms
        self.elastic = elastic
        self.normalize = normalize

    def __call__(self, img, target):

        assert isinstance(img, np.ndarray), 'Image data must be either numpy array'
        assert isinstance(target, np.ndarray), 'Label must be either numpy array'

        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7

        img = np.squeeze(img)
        img = np.moveaxis(img, 0, 2)
        img = self.transforms(img)
        img = torch.unsqueeze(img, 0)

        random.seed(seed) # apply this seed to target tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7

        target = np.squeeze(target)
        target = np.moveaxis(target, 0, 2)
        target = self.transforms(target)
        target = torch.unsqueeze(target, 0)

        # Random Elastic Transform
        if self.elastic:
            alpha = random.randint(0,2)
            preprocess = RandomElastic(alpha=alpha, sigma=0.06)
            img, target = preprocess(img, target)

        if self.normalize is not None:
            mean, std = self.normalize
            normarlize_transform = T.Normalize(mean, std)
            img = normarlize_transform(img)
        
        return img, target