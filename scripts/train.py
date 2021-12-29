
import json
import argparse
import os

from torch.utils.data import DataLoader
from dataset.dataloader import get_dataset
from model.trainer import UNetTrainer
from model.logger import Logger
from model.utils import multiclass_dice_coeff, Combined_CE_DC_Loss, dice_loss
from model.unet import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', type=str)
parser.add_argument('--arch', '-a', default='UNet', type=str)
parser.add_argument('--dropout_chance', '-d', default=0.0, type=float)
parser.add_argument('--load_weights', '-l', action='store_true')
parser.add_argument('--path', '-p', type=str)

args = parser.parse_args()

net_name = args.model

#train, valid = get_dataset("/home/mberezny/Downloads/leftImg8bit_trainvaltest/leftImg8bit")
#train, valid = get_dataset("/storage/brno2/home/kiiroq/leftImg8bit_trainvaltest/leftImg8bit")
train, valid = get_dataset(args.path)

train_loader = DataLoader(train, batch_size=16, num_workers=4, shuffle=True)
valid_loader = DataLoader(valid, batch_size=16, num_workers=4)

val_iter = len(valid)
train_iter = len(train)

json_path = os.path.join('../pretrained_weights', net_name+'.json')

archs = {'UNet': UNet,
        'NestedUNet': NestedUNet,
        'R2UNet': R2UNet,
        'AttUNet': AttUNet,
        'AttR2UNet': AttR2UNet}

if args.arch not in archs:
    print("Invalid architecture passed in parameters, defaulting to U-Net")
    args.arch = 'UNet'

if os.path.exists(json_path):
    with open(json_path) as fp:
        json_dict = json.load(fp)
else:
    json_dict = {'max_epoch': 200,
                 'epoch': 0,
                 'index': 0,
                 'train_iter': train_iter,
                 'val_iter': val_iter,
                 'best_loss': 100000,
                 'dropout_chance': args.dropout_chance,
                 'arch': args.arch,
                 'epochs_no_improvement': 0,
                 'patience': 100}
    with open(json_path, 'w') as fp:
        json.dump(json_dict, fp, sort_keys=True, indent=4)


logger = Logger(folder='../pretrained_weights',
                net_name=net_name,
                json_dict=json_dict)
logger.log_sample_size(train, valid)

loss = dice_loss #Combined_CE_DC_Loss()

trainer = UNetTrainer(start_epoch=json_dict['epoch'],
                      end_epoch=json_dict['max_epoch'],
                      criterion=loss,
                      logger=logger,
                      model_name=args.model,
                      dropout_chance=json_dict['dropout_chance'],
                      arch=archs[json_dict['arch']],
                      metric=multiclass_dice_coeff,
                      load=args.load_weights,
                      learning_rate=0.0002,
                      in_channels=3,
                      out_classes=8)

trainer.fit(train_loader, valid_loader)