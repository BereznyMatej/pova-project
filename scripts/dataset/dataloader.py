import torch
import os
import copy
import json
import numpy as np
import torchvision.transforms.functional as TF


from PIL import Image
from torch.utils.data import Dataset
from dataset.augmentations import get_augmentations

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
            7: 1,  # road
            8: 2,  # sidewalk
            9: 0,  # parking
            10: 0,  # rail track
            11: 3,  # building
            12: 4,  # wall
            13: 5,  # fence
            14: 0,  # guard rail
            15: 0,  # bridge
            16: 0,  # tunnel
            17: 6,  # pole
            18: 0,  # polegroup
            19: 7,  # traffic light
            20: 8,  # traffic sign
            21: 9,  # vegetation
            22: 10,  # terrain
            23: 11,  # sky
            24: 12,  # person
            25: 13,  # rider
            26: 14,  # car
            27: 15,  # truck
            28: 16,  # bus
            29: 0,  # caravan
            30: 0,  # trailer
            31: 17,  # train
            32: 18,  # motorcycle
            33: 19,  # bicycle
            -1: 0  # licenseplate
        }
        self.data, self.labels = self.load(data_path)


    def __mask_to_map__(self, label):
        mask = torch.zeros((label.size()[0], label.size()[1]), dtype=torch.uint8)
        for k in self.mapping:
            mask[label == k] = self.mapping[k]
        return mask


    def __process_image__(self, image_path, img_type, interpolation, label=False):
        
        pad = 10 if self.transform is not None else 0

        image = Image.open(image_path).convert(img_type)
        image = TF.resize(image, size=(128+pad, 256+pad), interpolation=interpolation)
        if label:
            image = torch.from_numpy(np.asarray(image, dtype=np.uint8))
            return self.__mask_to_map__(image)
        
        return TF.pil_to_tensor(image)


    def load(self, data_path):
        
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

        if self.transform is not None:
            data, label = self.transform((data, label))

        return {'x': data.float(), 'y': label.long()}

    def __len__(self):
        return self.data.size()[0]


def get_dataset(path):

    augmenations = get_augmentations((128,256), 0.5)

    train = Images(data_path=os.path.join(path, 'train'), transform=augmenations)
    valid = Images(data_path=os.path.join(path, 'val'))
    
    return train, valid
