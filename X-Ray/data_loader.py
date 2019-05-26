import numpy as np
import pandas as pd
import os
from glob import glob

import random
import numpy.random

import scipy.io
import numpy as np

from PIL import Image

import torch
import torch.utils.data as data_utils
import torchvision.transforms as transforms
import torchvision.transforms.functional
import matplotlib.pyplot as plt


def Xray_Dataset(path, train):
    if train:
        path += 'train/'
    else:
        path += 'test/'

    transforms_list = [
        ]

    data_transforms = transforms.Compose(
        [transforms.Resize((128, 128)),
         transforms.Grayscale(),
         #transforms.RandomCrop(128),
         transforms.ColorJitter(contrast=(1.7, 1.7)),
         #transforms.CenterCrop(128),
         transforms.ToTensor()
         ])

    image_dataset = torchvision.datasets.ImageFolder(path,data_transforms)

    return image_dataset

if __name__ == "__main__":
    train_loader = data_utils.DataLoader(
        Xray_Dataset('chest-xray-pneumonia/chest_xray/', train=True),
        batch_size=100,
        shuffle=False, num_workers=2)

    test_loader = data_utils.DataLoader(
        Xray_Dataset('chest-xray-pneumonia/chest_xray/', train=False),
        batch_size=100,
        shuffle=False, num_workers=2)

    for i, (data, label) in enumerate(train_loader):
        print(data.size())
        plt.imshow(data[0].squeeze().data.numpy())
        plt.show()
        print(label[0])
        break
