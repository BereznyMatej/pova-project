import torch
import os
import copy
import json
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class Images(Dataset):

    def __init__(self, transform=None, data_path=None):
        self.transform = transform
        self.mapping = {
            0: 0,  # unlabeled
            1: 0,  # ego vehicle
            2: 0,  # rect border
            3: 0,  # out of roi
            4: 0,  # static
            5: 0,  # dynamic
            6: 0,  # ground
            7: 4,  # road
            8: 0,  # sidewalk
            9: 0,  # parking
            10: 0,  # rail track
            11: 0,  # building
            12: 0,  # wall
            13: 0,  # fence
            14: 0,  # guard rail
            15: 0,  # bridge
            16: 0,  # tunnel
            17: 0,  # pole
            18: 0,  # polegroup
            19: 0,  # traffic light
            20: 0,  # traffic sign
            21: 0,  # vegetation
            22: 0,  # terrain
            23: 0,  # sky
            24: 1,  # person
            25: 0,  # rider
            26: 2,  # car
            27: 3,  # truck
            28: 0,  # bus
            29: 0,  # caravan
            30: 0,  # trailer
            31: 0,  # train
            32: 0,  # motorcycle
            33: 0,  # bicycle
            -1: 0  # licenseplate
        }
        self.data, self.labels = self.load(data_path)


    def __mask_to_map__(self, label):
        mask = torch.zeros((label.size()[0], label.size()[1]), dtype=torch.uint8)
        for k in self.mapping:
            mask[label == k] = self.mapping[k]
        return mask


    def __process_image__(self, image_path, img_type, interpolation, label=False):

        image = Image.open(image_path).convert(img_type)
        image = TF.resize(image, size=(128, 256), interpolation=interpolation)
        if label:
            image = torch.from_numpy(np.asarray(image, dtype=np.uint8))
            return self.__mask_to_map__(image)
        
        return TF.pil_to_tensor(image)


    def load(self, data_path, label=False):
        
        data = []
        labels = []


        for dir_name in os.listdir(data_path):
            print(f"Processing directory {dir_name}...", end=' ')
            dir_path = os.path.join(data_path, dir_name)
            label_list = os.listdir(os.path.join(dir_path, dir_name))
            
            if os.path.isfile(dir_path):
                continue 

            for img_path_name in os.listdir(dir_path):
                img_path = os.path.join(dir_path, img_path_name)
                if os.path.isfile(img_path):
                    img_name = img_path_name[:img_path_name.rfind('_')]
                    matching_labels = [s for s in label_list if img_name in s]
                    label_name = [s for s in matching_labels if 'labelIds' in s]
                    
                    if not label_name:
                        print(f"Missing label for {img_name}")
                        continue
                    label_path = os.path.join(dir_path, dir_name, label_name[0])

                    label = self.__process_image__(label_path, 'L',TF.InterpolationMode.NEAREST, label=True)
                    img = self.__process_image__(img_path, 'RGB', TF.InterpolationMode.BILINEAR)
                    data.append(img)
                    labels.append(label)

            print("Done.")

        return  torch.stack(data), torch.stack(labels)

    def __getitem__(self, index):        
        data = copy.deepcopy(self.data[index])
        label = copy.deepcopy(self.labels[index])
        
        return {'x': data.float(), 'y': label.long()}

    def __len__(self):
        return self.data.size()[0]


def get_dataset(path):

    train = Images(data_path=os.path.join(path, 'train'))
    valid = Images(data_path=os.path.join(path, 'val'))
    
    return train, valid
