import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, act):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.b_norm = nn.BatchNorm2d(out_channels)

        if act == 'relu':
            self.activation = nn.ReLU(out_channels)
        elif act == 'lrelu':
            self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.activation(self.b_norm(self.conv(x)))


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, act):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(ConvBlock(in_channels, out_channels, act),
                                         ConvBlock(out_channels, out_channels, act))

    def forward(self, x):
        return self.double_conv(x)


class DownTransition(nn.Module):

    def __init__(self, in_channels, out_channels, depth, act):
        super(DownTransition, self).__init__()
        self.ops = DoubleConv(in_channels, out_channels, act)
        self.depth = depth
        self.maxpool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d(p=0.5)

    def forward(self, x):
        out = self.ops(x)
        out_after_pool = out if self.depth == 3 else self.maxpool(out)
        return out_after_pool, self.dropout(out)


class UpTransition(nn.Module):

    def __init__(self, in_channels, out_channels, act):
        super(UpTransition, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        #self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.ops = DoubleConv(in_channels, out_channels, act)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.ops(x)

class OutTransition(nn.Module):

    def __init__(self, in_channels, out_classes):
        super(OutTransition, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_classes, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)