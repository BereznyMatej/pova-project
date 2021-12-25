from model.blocks import *

import torch
import torch.nn as nn
from PIL import Image
import numpy as np


class R2UNet(nn.Module):

    def __init__(self, in_channels, out_classes, act='relu', dropout_chance=0.0):
        super().__init__()
        
        channels = [64, 128, 256, 512, 1024]

        self.rcnn1 = DownTransition(layer_args={'in_channels': in_channels, 
                                                'out_channels': channels[0],
                                                'act': act},
                                    layer=RCNNBlock,
                                    dropout_chance=dropout_chance)
        self.rcnn2 = DownTransition(layer_args={'in_channels': channels[0],
                                                'out_channels': channels[1],
                                                'act': act},
                                    layer=RCNNBlock,
                                    dropout_chance=dropout_chance)
        self.rcnn3 = DownTransition(layer_args={'in_channels': channels[1],
                                                'out_channels': channels[2],
                                                'act': act},
                                    layer=RCNNBlock,
                                    dropout_chance=dropout_chance)
        self.rcnn4 = DownTransition(layer_args={'in_channels': channels[2],
                                                'out_channels': channels[3],
                                                'act': act},
                                    layer=RCNNBlock,
                                    dropout_chance=dropout_chance)
        self.rcnn5 = DownTransition(layer_args={'in_channels': channels[3],
                                                'out_channels': channels[4],
                                                'act': act},
                                    layer=RCNNBlock,
                                    apply_pool=False,
                                    dropout_chance=dropout_chance)

        self.up_rcnn5 = UpTransition(layer_args={'in_channels': channels[4],
                                                 'out_channels': channels[3], 
                                                 'act': act},
                                     layer=RCNNBlock)
        self.up_rcnn4 = UpTransition(layer_args={'in_channels': channels[3],
                                                 'out_channels': channels[2], 
                                                 'act': act},
                                     layer=RCNNBlock)
        self.up_rcnn3 = UpTransition(layer_args={'in_channels': channels[2],
                                                 'out_channels': channels[1], 
                                                 'act': act},
                                     layer=RCNNBlock)
        self.up_rcnn2 = UpTransition(layer_args={'in_channels': channels[1],
                                                 'out_channels': channels[0], 
                                                 'act': act},
                                     layer=RCNNBlock)
        self.out = OutTransition(channels[0], out_classes)

    def forward(self, x):
        x1, x1_drop = self.rcnn1(x)
        x2, x2_drop = self.rcnn2(x1)
        x3, x3_drop = self.rcnn3(x2)
        x4, x4_drop = self.rcnn4(x3)
        x5, _ = self.rcnn5(x4)

        d4 = self.up_rcnn5([x5, x4_drop])
        d3 = self.up_rcnn4([d4, x3_drop])
        d2 = self.up_rcnn3([d3, x2_drop])
        d1 = self.up_rcnn2([d2, x1_drop])

        out = self.out(d1)

        return [out]


class UNet(nn.Module):
    
    def __init__(self, in_channels, out_classes, act='relu', dropout_chance=0.0):
        super(UNet, self).__init__()
        channels = [64, 128, 256, 512, 1024]

        self.down_tr64 =   DownTransition(layer_args={'in_channels': in_channels, 
                                                      'out_channels': channels[0],
                                                      'act': act},
                                          layer=DoubleConv,
                                          dropout_chance=dropout_chance)
        self.down_tr128 =  DownTransition(layer_args={'in_channels': channels[0],
                                                      'out_channels': channels[1],
                                                      'act': act},
                                          layer=DoubleConv,
                                          dropout_chance=dropout_chance)
        self.down_tr256 =  DownTransition(layer_args={'in_channels': channels[1],
                                                      'out_channels': channels[2],
                                                      'act': act},
                                          layer=DoubleConv,
                                          dropout_chance=dropout_chance)
        self.down_tr512 =  DownTransition(layer_args={'in_channels': channels[2],
                                                      'out_channels': channels[3],
                                                      'act': act},
                                          layer=DoubleConv,
                                          dropout_chance=dropout_chance)
        self.down_tr1024 = DownTransition(layer_args={'in_channels': channels[3],
                                                      'out_channels': channels[4],
                                                      'act': act},
                                          layer=DoubleConv,
                                          apply_pool=False,
                                          dropout_chance=dropout_chance)

        self.up_tr512 = UpTransition(layer_args={'in_channels': channels[4],
                                                 'out_channels': channels[3], 
                                                 'act': act},
                                     layer=DoubleConv)
        self.up_tr256 = UpTransition(layer_args={'in_channels': channels[3],
                                                 'out_channels': channels[2], 
                                                 'act': act},
                                     layer=DoubleConv)
        self.up_tr128 = UpTransition(layer_args={'in_channels': channels[2],
                                                 'out_channels': channels[1], 
                                                 'act': act},
                                     layer=DoubleConv)
        self.up_tr64  = UpTransition(layer_args={'in_channels': channels[1],
                                                 'out_channels': channels[0], 
                                                 'act': act},
                                     layer=DoubleConv)
        self.out = OutTransition(channels[0], out_classes)
    

    def forward(self, x):
        down64, skip_down64 = self.down_tr64(x)
        down128, skip_down128 = self.down_tr128(down64)
        down256, skip_down256 = self.down_tr256(down128)
        down512, skip_down512 = self.down_tr512(down256)
        down1024, _ = self.down_tr1024(down512)

        up512 = self.up_tr512([down1024, skip_down512])
        up256 = self.up_tr256([up512, skip_down256])
        up128 = self.up_tr128([up256, skip_down128])
        up64 = self.up_tr64([up128, skip_down64])
        out = self.out(up64)

        return [out]


class NestedUNet(nn.Module):

    def __init__(self, in_channels, out_classes, act='relu',
                 dropout_chance=0.0):
        super(NestedUNet, self).__init__()

        channels = [32, 64, 128, 256, 512]
        
        self.down_0 = DownTransition(layer_args={'in_channels': in_channels, 
                                                 'out_channels': channels[0],
                                                  'act': act},
                                     layer=DoubleConv,
                                     dropout_chance=dropout_chance)
        self.down_1 = DownTransition(layer_args={'in_channels': channels[0],
                                                 'out_channels': channels[1],
                                                 'act': act},
                                     layer=DoubleConv,
                                     dropout_chance=dropout_chance)
        self.down_2 = DownTransition(layer_args={'in_channels': channels[1],
                                                 'out_channels': channels[2],
                                                 'act': act},
                                     layer=DoubleConv,
                                     dropout_chance=dropout_chance)
        self.down_3 = DownTransition(layer_args={'in_channels': channels[2],
                                                 'out_channels': channels[3],
                                                 'act': act},
                                     layer=DoubleConv,
                                     dropout_chance=dropout_chance)
        self.down_4 = DownTransition(layer_args={'in_channels': channels[3],
                                                 'out_channels': channels[4],
                                                 'act': act},
                                     layer=DoubleConv,
                                     dropout_chance=dropout_chance,
                                     apply_pool=False)

        self.up_01 = UpTransition(layer_args={'in_channels': channels[1],
                                              'middle_channels': channels[1],
                                              'out_channels': channels[0], 
                                              'act': act},
                                  layer=DoubleNestedConv)
        self.up_11 = UpTransition(layer_args={'in_channels': channels[2],
                                              'middle_channels': channels[2],
                                              'out_channels': channels[1], 
                                              'act': act},
                                  layer=DoubleNestedConv)
        self.up_21 = UpTransition(layer_args={'in_channels': channels[3],
                                              'middle_channels': channels[3],
                                              'out_channels': channels[2], 
                                              'act': act},
                                  layer=DoubleNestedConv)
        self.up_31 = UpTransition(layer_args={'in_channels': channels[4],
                                              'middle_channels': channels[4],
                                              'out_channels': channels[3], 
                                              'act': act},
                                  layer=DoubleNestedConv)

        self.up_02 = UpTransition(layer_args={'in_channels': channels[1],
                                              'middle_channels': channels[1] + channels[0],
                                              'out_channels': channels[0], 
                                              'act': act},
                                  layer=DoubleNestedConv)
        self.up_12 = UpTransition(layer_args={'in_channels': channels[2],
                                              'middle_channels': channels[2] + channels[1],
                                              'out_channels': channels[1], 
                                              'act': act},
                                  layer=DoubleNestedConv)
        self.up_22 = UpTransition(layer_args={'in_channels': channels[3],
                                              'middle_channels': channels[3] + channels[2],
                                              'out_channels': channels[2], 
                                              'act': act},
                                  layer=DoubleNestedConv)

        self.up_03 = UpTransition(layer_args={'in_channels': channels[1],
                                              'middle_channels': channels[1] + channels[0]*2,
                                              'out_channels': channels[0], 
                                              'act': act},
                                  layer=DoubleNestedConv)
        self.up_13 = UpTransition(layer_args={'in_channels': channels[2],
                                              'middle_channels': channels[2] + channels[1]*2,
                                              'out_channels': channels[1], 
                                              'act': act},
                                  layer=DoubleNestedConv)

        self.up_04 = UpTransition(layer_args={'in_channels': channels[1],
                                              'middle_channels': channels[1] + channels[0]*3,
                                              'out_channels': channels[0], 
                                              'act': act},
                                  layer=DoubleNestedConv)
        
        self.out_tr1 = OutTransition(channels[0], out_classes)
        self.out_tr2 = OutTransition(channels[0], out_classes)
        self.out_tr3 = OutTransition(channels[0], out_classes)
        self.out_tr4 = OutTransition(channels[0], out_classes)


    def __call__(self, x):
        
        x0, x0_drop = self.down_0(x)
        x1, x1_drop = self.down_1(x0)
        x01 = self.up_01([x1_drop, x0_drop])

        x2, x2_drop = self.down_2(x1)
        x11 = self.up_11([x2_drop, x1_drop])
        x02 = self.up_02([x11, x01, x0_drop])

        x3, x3_drop = self.down_3(x2)
        x21 = self.up_21([x3_drop, x2_drop])
        x12 = self.up_12([x21, x11, x1_drop])
        x03 = self.up_03([x12, x02, x01, x0_drop])

        x4, x4_drop = self.down_4(x3)
        x31 = self.up_31([x4_drop, x3_drop])
        x22 = self.up_22([x31, x21, x2_drop])
        x13 = self.up_13([x22, x12, x11, x1_drop])
        x04 = self.up_04([x13, x03, x02, x01, x0_drop])
        
        return [self.out_tr1(x01), self.out_tr2(x02), self.out_tr3(x03), self.out_tr4(x04)]
   

class AttUNet(nn.Module):
    
    def __init__(self, in_channels, out_classes, act='relu', dropout_chance=0.0):
        super(AttUNet, self).__init__()
        channels = [64, 128, 256, 512, 1024]

        self.down_tr64 =   DownTransition(layer_args={'in_channels': in_channels, 
                                                      'out_channels': channels[0],
                                                      'act': act},
                                          layer=DoubleConv,
                                          dropout_chance=dropout_chance)
        self.down_tr128 =  DownTransition(layer_args={'in_channels': channels[0],
                                                      'out_channels': channels[1],
                                                      'act': act},
                                          layer=DoubleConv,
                                          dropout_chance=dropout_chance)
        self.down_tr256 =  DownTransition(layer_args={'in_channels': channels[1],
                                                      'out_channels': channels[2],
                                                      'act': act},
                                          layer=DoubleConv,
                                          dropout_chance=dropout_chance)
        self.down_tr512 =  DownTransition(layer_args={'in_channels': channels[2],
                                                      'out_channels': channels[3],
                                                      'act': act},
                                          layer=DoubleConv,
                                          dropout_chance=dropout_chance)
        self.down_tr1024 = DownTransition(layer_args={'in_channels': channels[3],
                                                      'out_channels': channels[4],
                                                      'act': act},
                                          layer=DoubleConv,
                                          apply_pool=False,
                                          dropout_chance=dropout_chance)

        self.up_tr512 = UpTransition(layer_args={'in_channels': channels[4],
                                                 'out_channels': channels[3], 
                                                 'act': act},
                                     layer=DoubleConv,
                                     attention=True)
        self.up_tr256 = UpTransition(layer_args={'in_channels': channels[3],
                                                 'out_channels': channels[2], 
                                                 'act': act},
                                     layer=DoubleConv,
                                     attention=True)
        self.up_tr128 = UpTransition(layer_args={'in_channels': channels[2],
                                                 'out_channels': channels[1], 
                                                 'act': act},
                                     layer=DoubleConv,
                                     attention=True)
        self.up_tr64  = UpTransition(layer_args={'in_channels': channels[1],
                                                 'out_channels': channels[0], 
                                                 'act': act},
                                     layer=DoubleConv,
                                     attention=True)
        self.out = OutTransition(channels[0], out_classes)


    def forward(self, x):
        down64, skip_down64 = self.down_tr64(x)
        down128, skip_down128 = self.down_tr128(down64)
        down256, skip_down256 = self.down_tr256(down128)
        down512, skip_down512 = self.down_tr512(down256)
        down1024, _ = self.down_tr1024(down512)

        up512 = self.up_tr512([down1024, skip_down512])
        up256 = self.up_tr256([up512, skip_down256])
        up128 = self.up_tr128([up256, skip_down128])
        up64 = self.up_tr64([up128, skip_down64])
        out = self.out(up64)

        return [out]


class AttR2UNet(nn.Module):

    def __init__(self, in_channels, out_classes, act='relu', dropout_chance=0.0):
        super().__init__()
        
        channels = [64, 128, 256, 512, 1024]

        self.rcnn1 = DownTransition(layer_args={'in_channels': in_channels, 
                                                'out_channels': channels[0],
                                                'act': act},
                                    layer=RCNNBlock,
                                    dropout_chance=dropout_chance)
        self.rcnn2 = DownTransition(layer_args={'in_channels': channels[0],
                                                'out_channels': channels[1],
                                                'act': act},
                                    layer=RCNNBlock,
                                    dropout_chance=dropout_chance)
        self.rcnn3 = DownTransition(layer_args={'in_channels': channels[1],
                                                'out_channels': channels[2],
                                                'act': act},
                                    layer=RCNNBlock,
                                    dropout_chance=dropout_chance)
        self.rcnn4 = DownTransition(layer_args={'in_channels': channels[2],
                                                'out_channels': channels[3],
                                                'act': act},
                                    layer=RCNNBlock,
                                    dropout_chance=dropout_chance)
        self.rcnn5 = DownTransition(layer_args={'in_channels': channels[3],
                                                'out_channels': channels[4],
                                                'act': act},
                                    layer=RCNNBlock,
                                    apply_pool=False,
                                    dropout_chance=dropout_chance)

        self.up_rcnn5 = UpTransition(layer_args={'in_channels': channels[4],
                                                 'out_channels': channels[3], 
                                                 'act': act},
                                     layer=RCNNBlock,
                                     attention=True)
        self.up_rcnn4 = UpTransition(layer_args={'in_channels': channels[3],
                                                 'out_channels': channels[2], 
                                                 'act': act},
                                     layer=RCNNBlock,
                                     attention=True)
        self.up_rcnn3 = UpTransition(layer_args={'in_channels': channels[2],
                                                 'out_channels': channels[1], 
                                                 'act': act},
                                     layer=RCNNBlock,
                                     attention=True)
        self.up_rcnn2 = UpTransition(layer_args={'in_channels': channels[1],
                                                 'out_channels': channels[0], 
                                                 'act': act},
                                     layer=RCNNBlock,
                                     attention=True)
        self.out = OutTransition(channels[0], out_classes)


    def forward(self, x):
        x1, x1_drop = self.rcnn1(x)
        x2, x2_drop = self.rcnn2(x1)
        x3, x3_drop = self.rcnn3(x2)
        x4, x4_drop = self.rcnn4(x3)
        x5, _ = self.rcnn5(x4)

        d4 = self.up_rcnn5([x5, x4_drop])
        d3 = self.up_rcnn4([d4, x3_drop])
        d2 = self.up_rcnn3([d3, x2_drop])
        d1 = self.up_rcnn2([d2, x1_drop])

        out = self.out(d1)

        return [out]


if __name__ == "__main__":
    model = AttR2UNet(3, 8)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = nn.DataParallel(model, device_ids = [i for i in range(torch.cuda.device_count())])
    model.to(device)
    model.train()
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
    #img = Image.open('/home/mberezny/git/pova-project/data/train/zurich_000117_000019_leftImg8bit.png')
    #img = torch.tensor(np.expand_dims(np.transpose(np.asarray(img), (2, 0, 1)), axis=0)).float().to(device)
    x = torch.rand((1,3,128,256))
    out = model(x)