import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

def dummy_act(x):
    return x

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, act, 
                kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, 
                              stride=stride,
                              padding=padding)
        self.b_norm = nn.BatchNorm2d(out_channels)
        if act == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif act == 'lrelu':
            self.activation = nn.LeakyReLU(inplace=True)
        elif act == None:
            self.activation = dummy_act
        elif act == 'sigmoid':
            self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.b_norm(self.conv(x)))


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, act):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(ConvBlock(in_channels, out_channels, act),
                                         ConvBlock(out_channels, out_channels, act))

    def forward(self, x):
        return self.double_conv(x)


class DoubleNestedConv(nn.Module):

    def __init__(self, in_channels, middle_channels, out_channels, act):
        super(DoubleNestedConv, self).__init__()
        self.double_conv = nn.Sequential(ConvBlock(middle_channels, out_channels, act),
                                         ConvBlock(out_channels, out_channels, act))

    def forward(self, x):
        return self.double_conv(x)


class DownTransition(nn.Module):

    def __init__(self, layer_args, apply_pool=True, layer=DoubleConv,
                 dropout_chance=0.0):
        super(DownTransition, self).__init__()
        self.ops = layer(**layer_args)
        self.apply_pool = apply_pool
        self.maxpool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d(p=dropout_chance)

    def forward(self, x):
        out = self.ops(x)
        out_after_pool = self.maxpool(out) if self.apply_pool else out
        return out_after_pool, self.dropout(out)


class UpTransition(nn.Module):

    def __init__(self, layer_args, layer=DoubleConv, attention=False):
        super(UpTransition, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels=layer_args['in_channels'],
                                     out_channels=layer_args['out_channels'],
                                     kernel_size=2, stride=2)
        self.ops = layer(**layer_args)
        
        if attention:
            self.attention = AttentionBlock(layer_args['out_channels'], layer_args['out_channels'], layer_args['out_channels']//2)
        else:
            self.attention = None
            
    def forward(self, x):
        x[0] = self.up(x[0])
        if self.attention is not None:
            x[0] = self.attention(x)
        x = torch.cat(x, dim=1)

        return self.ops(x)

class OutTransition(nn.Module):

    def __init__(self, in_channels, out_classes):
        super(OutTransition, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_classes, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)


class RCNNBlock(nn.Module):

    def __init__(self, in_channels, out_channels, act, t=2):
        super().__init__()
        self.rcnn = nn.Sequential(RecurrentBlock(out_channels, t, act),
                                  RecurrentBlock(out_channels, t, act))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        x = self.conv(x)
        x1 = self.rcnn(x)
        return x + x1


class AttentionBlock(nn.Module):
    
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = ConvBlock(F_g, F_int, kernel_size=1, padding=0, act=None)
        self.W_x = ConvBlock(F_l, F_int, kernel_size=1, padding=0, act=None)
        self.psi = ConvBlock(F_int, 1, kernel_size=1, padding=0, act='sigmoid')
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        g1 = self.W_g(x[1])
        x1 = self.W_x(x[0])
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x[0]*psi

class RecurrentBlock(nn.Module):
    
    def __init__(self, out_channels, t, act):
        super().__init__()
        self.t = t
        self.conv = ConvBlock(out_channels, out_channels, act=act)

    def forward(self, x):
        x1 = self.conv(x)
        for i in range(1, self.t):
           x1 = self.conv(x+x1)
        return x1