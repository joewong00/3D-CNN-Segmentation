from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from utils.utils import convert_to_numpy, add_channel, to_depth_last

import numbers
import numpy as np
import torch
import torchvision.transforms as T

class RandomElastic(object):
    """Random Elastic transformation by CV2 method on image by alpha, sigma parameter.
    Reference: https://github.com/gatsby2016/Augmentation-PyTorch-Transforms.git
        # you can refer to:  https://blog.csdn.net/qq_27261889/article/details/80720359
        # https://blog.csdn.net/maliang_1993/article/details/82020596
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html#scipy.ndimage.map_coordinates
    Args:
        alpha (float): alpha value for Elastic transformation, factor
        if alpha is 0, output is original whatever the sigma;
        if alpha is 1, output only depends on sigma parameter;
        if alpha < 1 or > 1, it zoom in or out the sigma's Relevant dx, dy.
        sigma (float): sigma value for Elastic transformation, should be \ in (0.05,0.1)

        img (numpy array) 3D:(C * D * W * H) 2D:(C * W * H)
        mask (numpy array) same size/shape as img, if not assign, set None.
       
    Output:
        transformed img (numpy array) 3D:(C * D * W * H) 2D:(C * W * H)
        transformed mask (numpy array) 3D:

    Example:
    preprocess = RandomElastic(alpha=2, sigma=0.06)
    transformed_data, transformed_target = preprocess(data, target) 
    """
    def __init__(self, alpha, sigma, to_tensor=True):
        assert isinstance(alpha, numbers.Number) and isinstance(sigma, numbers.Number), \
            "alpha and sigma should be a single number."
        assert 0.05 <= sigma <= 0.1, \
            "In pathological image, sigma should be in (0.05,0.1)"

        self.alpha = alpha
        self.sigma = sigma
        self.to_tensor = to_tensor

    @staticmethod
    def RandomElasticCV2(img, alpha, sigma, mask=None, to_tensor=True):
        alpha = img.shape[1] * alpha
        sigma = img.shape[1] * sigma
        if mask is not None:
            mask = np.array(mask).astype(np.uint8)
            img = np.concatenate((img, mask), axis=3)

        # W * H * D * C
        shape = img.shape
        depth = shape[2]

        # W * H * C
        shape = (shape[0], shape[1], shape[3])

        transformed = []

        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha

        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

        # Across the depth axis
        for i in range(depth):
            transformed_img = map_coordinates(img[:,:,i,:], indices, order=0, mode='reflect').reshape(shape)
            transformed.append(transformed_img)

        img = np.array(transformed)

        # Move channel to dimension1
        img = np.moveaxis(img,3,0)

        if to_tensor:
            img = torch.from_numpy(img)
        
        if mask is not None:
            return img[:1, ...], img[1:,...]
        else:
            return img

    def __call__(self, img, mask=None):

        img = convert_to_numpy(img)
        img = to_depth_last(img)
        img = add_channel(img,3)

        if mask is not None:
            mask = convert_to_numpy(mask)
            mask = to_depth_last(mask)
            mask = add_channel(mask,3)

        return self.RandomElasticCV2(np.array(img), self.alpha, self.sigma, mask, to_tensor=self.to_tensor)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(alpha value={0})'.format(self.alpha)
        format_string += ', sigma={0}'.format(self.sigma)
        format_string += ')'
        return format_string
