import numpy as np
import argparse
import matplotlib.pyplot as plt

from dataset.dataloader import Images
from torch.utils.data import DataLoader
from model.trainer import UNetTrainer
from PIL import Image
from model.utils import multiclass_dice_coeff


parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', type=str)

args = parser.parse_args()

test = Images(data_path="/home/mberezny/Downloads/leftImg8bit_trainvaltest/leftImg8bit/test")
data_loader = DataLoader(test, batch_size=1, num_workers=4)

model = UNetTrainer(load=True,
                    model_name=args.model,
                    in_channels=3,
                    out_classes=8,
                    metric=multiclass_dice_coeff
                   )

results, seg_evals = model.infer(data_loader)


for i in range(36,40):
    random_sample = results[i]
    random_sample = np.argmax(random_sample[0], axis=0)*30
    plt.imsave(f"kekw{i}.png", random_sample, cmap='Greys')

seg_evals = np.array(seg_evals)
print(seg_evals)
seg_evals[seg_evals == 0.0] = np.nan
print("Dice similarity coefficient is: {:.2%}".format(np.nanmean(seg_evals)))


#random_sample = np.squeeze(random_sample*30, axis=0)
#img = Image.fromarray(random_sample)
#img.save('kekw.png')