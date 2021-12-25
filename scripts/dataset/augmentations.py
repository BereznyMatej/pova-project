from os import XATTR_CREATE
from numpy.random.mtrand import rand
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as AUG
import torchvision.transforms.functional as TF
import random
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


GLOBAL_RANDOM_STATE = np.random.RandomState(69)


class Mirroring:

    def __init__(self, random_state, p):
        self.random_state = random_state
        self.axes = (0,1)
        self.prob = p 


    def __call__(self, data):
        x, y = data
        if 1 in self.axes and self.random_state.uniform() < self.prob:
            x = torch.flip(x, [2])
            y = torch.flip(y, [1])

        return (x, y)


class Crop:

    def __init__(self, size=(128, 256)):
        self.size = size


    def __call__(self, data):
        x, y = data
        i, j, h, w = AUG.RandomCrop.get_params(x, output_size=self.size)
        x = TF.crop(x, i, j, h, w)
        y = TF.crop(y, i, j, h, w)
        
        return (x, y)


class ElasticTransform:
    """Based on implimentation on
    https://gist.github.com/erniejunior/601cdf56d2b424757de5"""

    def __init__(self, random_state, alpha=2000, sigma=50, execution_probability=0.3,
                 **kwargs):
        """
        :param spline_order: the order of spline interpolation (use 0 for labeled images)
        :param alpha: scaling factor for deformations
        :param sigma: smoothing factor for Gaussian filter
        :param execution_probability: probability of executing this transform
        :param apply_3d: if True apply deformations in each axis
        """
        self.random_state = random_state
        self.alpha = alpha
        self.sigma = sigma
        self.execution_probability = execution_probability


    def __call__(self, data):
        img, label = data
        if self.random_state.uniform() < self.execution_probability:        
            assert img.ndim in [2, 3]

            volume_shape = img[0].shape

            dy, dx = [
                gaussian_filter(
                    self.random_state.randn(*volume_shape),
                    self.sigma, mode="reflect"
                ) * self.alpha for _ in range(2)
            ]

            
            y_dim, x_dim = volume_shape
            y, x = np.meshgrid(np.arange(y_dim), np.arange(x_dim), indexing='ij')
            indices = y + dy, x + dx

            label = torch.from_numpy(map_coordinates(label, indices, order=0, mode='reflect'))
            channels = [map_coordinates(c, indices, order=3, mode='reflect') for c in img]
            img =  torch.from_numpy(np.stack(channels, axis=0))

        return (img, label)


class GammaCorrection:

    def __init__(self, random_state, gamma=0.3, p=0.5) -> None:
        self.gamma = gamma
        self.p = p
        self.random_state = random_state

    def __call__(self, data):
        x, y = data
        
        if self.random_state.uniform() < self.p:
            x = TF.adjust_gamma(x, gamma=self.gamma)
        
        return (x, y)


class Contrast:
    
    def __init__(self, random_state, p=0.5):
        self.random_state = random_state
        self.p = p

    def __call__(self, data) -> None:
        
        x, y = data
        if self.random_state.uniform() < self.p:
            x = TF.adjust_contrast(x, contrast_factor=self.random_state.uniform())
        
        return (x, y)


class Saturation:

    def __init__(self, random_state, p=0.5):
        self.random_state = random_state
        self.p = p

    
    def __call__(self, data):

        x, y = data
        if self.random_state.uniform() < self.p:
            x = TF.adjust_saturation(x, saturation_factor=self.random_state.uniform(0, 2))

        return (x, y)


class ColorTransform:

    def __init__(self, random_state, p=0.5) -> None:
        self.random_state = random_state
        self.p = p
    
    def __call__(self, data):
        
        x, y = data
        if self.random_state.uniform() < self.p:
            hue_factor = self.random_state.uniform(low=-0.5, high=0.5)
            x = TF.adjust_hue(x, hue_factor=hue_factor)
        return (x, y)


def get_augmentations(size, probability):
    
    random_state = np.random.RandomState(GLOBAL_RANDOM_STATE.randint(10000000))

    return AUG.Compose([Crop(size),
                        Mirroring(random_state, probability),
                        ColorTransform(random_state, p=probability),
                        Saturation(random_state, p=probability),
                        Contrast(random_state=random_state, p=probability),
                        GammaCorrection(random_state=random_state, p=probability),
                        ElasticTransform(random_state=random_state, p=probability)])