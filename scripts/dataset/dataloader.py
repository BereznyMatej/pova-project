import torch
import os
import copy
import json
import numpy as np
import torchvision.transforms.functional as TF
from torch.nn.functional import normalize
import threading

from PIL import Image
from torch.utils.data import Dataset
from dataset.augmentations import get_augmentations


class Images(Dataset):

    rgb_mapping = {
     0: (0, 0, 0),
     1: (203, 195, 227),
     2: (144,238,144),
     3: (255, 0, 0),
     4: (135,206,235),
     5: (139,69,19),
     6: (0, 0, 255),
     7: (255,165,0),
    }

    def __init__(self, transform=None, data_path=None, mean=(73.4614, 83.2210, 72.7894), std=(46.2917, 46.9659, 46.3322)):
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
            8: 1,  # sidewalk
            9: 0,  # parking
            10: 1,  # rail track
            11: 5,  # building
            12: 5,  # wall
            13: 5,  # fence
            14: 5,  # guard rail
            15: 5,  # bridge
            16: 5,  # tunnel
            17: 3,  # pole
            18: 3,  # polegroup
            19: 3,  # traffic light
            20: 3,  # traffic sign
            21: 2,  # vegetation
            22: 2,  # terrain
            23: 4,  # sky
            24: 6,  # person
            25: 6,  # rider
            26: 7,  # car
            27: 7,  # truck
            28: 7,  # bus
            29: 7,  # caravan
            30: 7,  # trailer
            31: 7,  # train
            32: 7,  # motorcycle
            33: 7,  # bicycle
            -1: 0  # licenseplate
        }
        self.mean = mean
        self.std = std
        self.data, self.labels = self.load(data_path)
        

    @classmethod
    def map_to_rgb(cls, mask):
        mask = mask.astype(np.uint8)
        rgbimg = np.zeros((3, mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for k in cls.rgb_mapping:
            rgbimg[0][mask == k] = cls.rgb_mapping[k][0]
            rgbimg[1][mask == k] = cls.rgb_mapping[k][1]
            rgbimg[2][mask == k] = cls.rgb_mapping[k][2]
                    
        return rgbimg

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
        else: 
            image = TF.pil_to_tensor(image)
            
            #chan = []
            #for c in range(image.shape[0]):
            #    chan.append((image[c] - self.mean[c]) / self.std[c])
            #image = torch.stack(chan)

            return image


    def __load(self, data_path, dir_name, idx, data, labels):
        dir_path = os.path.join(data_path, dir_name)
        label_list = os.listdir(os.path.join(dir_path, dir_name))
            
        if os.path.isfile(dir_path):
            return

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
                data[idx].append(img)
                labels[idx].append(label)
        print(f"Processed directory {dir_name}...")


    def load(self, data_path):
        
        dir_list = os.listdir(data_path)

        data = [[] for _ in dir_list]
        labels = [[] for _ in dir_list]

        threads = []

        for idx, dir_name in enumerate(os.listdir(data_path)):
            threads.append(threading.Thread(target=self.__load, args=(data_path, dir_name, idx, data, labels)))
            threads[idx].start()

        for t in threads:
            t.join()
        
        return  torch.stack(self.__flatten(data)), torch.stack(self.__flatten(labels))


    def __flatten(self, data):
        return [item for sublist in data for item in sublist]


    def __getitem__(self, index):



        if self.transform is not None:
            data = copy.deepcopy(self.data[index])
            label = copy.deepcopy(self.labels[index])
            data, label = self.transform((data, label))
        else:
            data = self.data[index]
            label = self.labels[index]
        return {'x': data.float(), 'y': label.long()}


    def __len__(self):
        return self.data.size()[0]


def get_dataset(path):

    augmenations = get_augmentations((128,256), 0.5)

    train = Images(data_path=os.path.join(path, 'train'), transform=augmenations)
    valid = Images(data_path=os.path.join(path, 'val'))

    return train, valid


class ImagesTest(Images):

    def __init__(self, transform=None, data_path=None, mean=(73.4614, 83.221, 72.7894), std=(46.2917, 46.9659, 46.3322)):
        super().__init__(transform=transform, data_path=data_path, mean=mean, std=std)
        self.mapping = {
            0: 0,  # unlabeled
            1: 0,  # ego vehicle
            2: 0,  # rect border
            3: 0,  # out of roi
            4: 0,  # static
            5: 0,  # dynamic
            6: 0,  # ground
            7: 1,  # road
            8: 1,  # sidewalk
            9: 0,  # parking
            10: 0,  # rail track
            11: 5,  # building
            12: 5,  # wall
            13: 5,  # fence
            14: 0,  # guard rail
            15: 0,  # bridge
            16: 0,  # tunnel
            17: 3,  # pole
            18: 0,  # polegroup
            19: 3,  # traffic light
            20: 3,  # traffic sign
            21: 2,  # vegetation
            22: 2,  # terrain
            23: 4,  # sky
            24: 6,  # person
            25: 6,  # rider
            26: 7,  # car
            27: 7,  # truck
            28: 7,  # bus
            29: 0,  # caravan
            30: 0,  # trailer
            31: 7,  # train
            32: 7,  # motorcycle
            33: 7,  # bicycle
            -1: 0  # licenseplate
        }