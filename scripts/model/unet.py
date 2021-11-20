from model.blocks import *

import torch
import torch.nn as nn
from PIL import Image
import numpy as np


class UNet(nn.Module):
    def __init__(self, in_channels, out_classes, act='relu'):
        super(UNet, self).__init__()

        self.down_tr64 = DownTransition(in_channels, 64, 0, act)
        self.down_tr128 = DownTransition(64, 128, 1, act)
        self.down_tr256 = DownTransition(128, 256, 2, act)
        self.down_tr512 = DownTransition(256, 512, 3, act)

        self.up_tr256 = UpTransition(512, 256, act)
        self.up_tr128 = UpTransition(256, 128, act)
        self.up_tr64 = UpTransition(128, 64, act)
        self.out_tr = OutTransition(64, out_classes)
    
    def forward(self, x):
        down64, skip_down64 = self.down_tr64(x)
        down128, skip_down128 = self.down_tr128(down64)
        down256, skip_down256 = self.down_tr256(down128)
        down512, skip_down512 = self.down_tr512(down256)

        up256 = self.up_tr256(down512, skip_down256)
        up128 = self.up_tr128(up256, skip_down128)
        up64 = self.up_tr64(up128, skip_down64)
        out = self.out_tr(up64)

        return out


if __name__ == "__main__":
    model = UNet(3, 30)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = nn.DataParallel(model, device_ids = [i for i in range(torch.cuda.device_count())])
    model.to(device)
    model.train()
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
    img = Image.open('/home/mberezny/git/pova-project/data/train/zurich_000117_000019_leftImg8bit.png')
    img = torch.tensor(np.expand_dims(np.transpose(np.asarray(img), (2, 0, 1)), axis=0)).float().to(device)
    x = torch.rand((1,3,2048,1024))
    out = model(x)