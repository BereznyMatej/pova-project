import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


class Combined_CE_DC_Loss:

    def __init__(self, weight=None):
        self.ce = nn.CrossEntropyLoss(wieght=weight)

    def __call__(self, input_tensor: Tensor, target: Tensor):    
        
        ce_loss = self.ce(input_tensor, target)
        dc_loss = dice_loss(input_tensor, target, multiclass=True)
 
        return ce_loss + dc_loss


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = True, epsilon=1e-6):
    # Average of Dice coefficient for all classes

    input = F.softmax(input, dim=1).float()
    target = F.one_hot(target, 8).permute(0, 3, 1, 2).float()
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = True):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    res = fn(input, target, reduce_batch_first=True)
    return 1 - res

