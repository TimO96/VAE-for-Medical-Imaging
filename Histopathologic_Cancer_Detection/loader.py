from model import *
from distributions import *

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader,Dataset
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
from torch.distributions.kl import kl_divergence
from conv_layer_architecture import *
from PIL import Image

class RandomRotate(object):
    """Rotate the given PIL.Image by either 0, 90, 180, 270."""

    def __call__(self, img):
        random_rotation = numpy.random.randint(4, size=1)
        if random_rotation == 0:
            pass
        else:
            img = torchvision.transforms.functional.rotate(img, random_rotation*90)
        return img

class CancerDataset(Dataset):
    def __init__(self, datafolder, datatype='train', domain_list=[], augmentation=False, transform=None, labels_dict={}):
        self.datafolder = datafolder
        self.datatype = datatype
        self.image_files_list = [s for s in os.listdir(datafolder)]
        self.transform = transform
        self.labels_dict = labels_dict
        if self.datatype == 'train':
            self.labels = [labels_dict[i.split('.')[0]] for i in self.image_files_list]
        else:
            self.labels = [0 for _ in range(len(self.image_files_list))]

        self.to_tensor = transforms.ToTensor()
        self.center_crop = transforms.CenterCrop(300)
        self.resize = transforms.Resize((64, 64), interpolation=Image.BILINEAR)
        self.hflip = transforms.RandomHorizontalFlip()
        self.vflip = transforms.RandomVerticalFlip()
        self.rrotate = RandomRotate()
        self.to_pil = transforms.ToPILImage()

    def __len__(self):
        return len(self.image_files_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.datafolder, self.image_files_list[idx])
        image = Image.open(img_name)
        image = self.transform(image)
        img_name_short = self.image_files_list[idx].split('.')[0]

        if self.datatype == 'train':
            label = self.labels_dict[img_name_short]
        else:
            label = 0
        return image, label
