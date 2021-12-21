import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as AUG
import torchvision.transforms.functional as TF

GLOBAL_RANDOM_STATE = np.random.RandomState(69)


class Mirroring:

    def __init__(self, random_state, p):
        self.random_state = random_state
        self.axes = (0,1)
        self.prob = p 


    def __call__(self, data):
        
        x, y = data

        #if 0 in self.axes and self.random_state.uniform() < self.prob:
        #    x = torch.flip(x, [1])
        #    y = torch.flip(y, [0])
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


def get_augmentations(size, probability):
    
    random_state = np.random.RandomState(GLOBAL_RANDOM_STATE.randint(10000000))

    return AUG.Compose([Crop(size),
                        Mirroring(random_state, probability),
                        ])