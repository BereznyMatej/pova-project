import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
import json
from model.unet import *


from dataset.dataloader import ImagesTest, Images
from torch.utils.data import DataLoader
from model.trainer import UNetTrainer
from model.utils import IoU

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', type=str)
parser.add_argument('--path', '-p', type=str)
parser.add_argument('--save_images', '-s', action='store_true')

args = parser.parse_args()

test = ImagesTest(data_path=f"{args.path}/val")
data_loader = DataLoader(test, batch_size=1, num_workers=4, shuffle=False)
iou = IoU(8, ignore_index=0)

archs = {'UNet': UNet,
        'NestedUNet': NestedUNet,
        'R2UNet': R2UNet,
        'AttUNet': AttUNet,
        'AttR2UNet': AttR2UNet}

json_path = os.path.join('../pretrained_weights', args.model+'.json')
if os.path.exists(json_path):
    with open(json_path) as fp:
        json_dict = json.load(fp)

class_names = ['background', 'flat', 'nature', 'object', 'sky', 'construction', 'human', 'vehicle']

model = UNetTrainer(load=True,
                    model_name=args.model,
                    in_channels=3,
                    out_classes=8,
                    metric=iou,
                    arch=archs[json_dict['arch']]
                   )

results = model.infer(data_loader)

if args.save_images:
    for idx, item in enumerate(results):
        sample =  np.argmax(item[0], axis=0)
        sample = Images.map_to_rgb(sample)
        plt.imsave(f"../seg_results/seg_{idx}.png", sample.transpose(1,2,0))


class_iou, mean_iou = iou.value()

print(f"Mean IoU for is {mean_iou}")
for idx, item in enumerate(class_iou):
    print(f"IoU for class {class_names[idx]} is {item}")

