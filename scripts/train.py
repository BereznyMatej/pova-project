
import json
import argparse
import os

from torch.utils.data import DataLoader
from dataset.dataloader import get_dataset
from model.trainer import UNetTrainer
from model.logger import Logger
from model.utils import multiclass_dice_coeff, ce_and_dc_loss

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', type=str)
parser.add_argument('--load_weights', '-l', action='store_true')

args = parser.parse_args()

net_name = args.model

train, valid = get_dataset("/home/mberezny/Downloads/leftImg8bit_trainvaltest/leftImg8bit")

train_loader = DataLoader(train, batch_size=4, num_workers=2, shuffle=True)
valid_loader = DataLoader(valid, batch_size=4, num_workers=2)

val_iter = len(valid)
train_iter = len(train)

json_path = os.path.join('../pretrained_weights', net_name+'.json')


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
                 'epochs_no_improvement': 0,
                 'patience': 100}
    with open(json_path, 'w') as fp:
        json.dump(json_dict, fp, sort_keys=True, indent=4)


logger = Logger(folder='../pretrained_weights',
                net_name=net_name,
                json_dict=json_dict)
logger.log_sample_size(train, valid)

loss = ce_and_dc_loss

trainer = UNetTrainer(start_epoch=json_dict['epoch'],
                      end_epoch=json_dict['max_epoch'],
                      criterion=loss,
                      metric=multiclass_dice_coeff,
                      logger=logger,
                      model_name=args.model,
                      momentum=0.95,
                      load=args.load_weights,
                      learning_rate=0.0002,
                      in_channels=3,
                      out_classes=8)

trainer.fit(train_loader, valid_loader)