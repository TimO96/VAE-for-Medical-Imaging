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


def Xray_Dataset(path):
    transforms_list = [
        ]

    data_transforms = transforms.Compose(
        [transforms.Resize((64, 64)),
         #transforms.RandomCrop(128),
         transforms.RandomHorizontalFlip(p=0.5),
         transforms.RandomVerticalFlip(p=0.5),
         transforms.ToTensor()
         ])

    image_dataset = torchvision.datasets.ImageFolder(path,data_transforms)

    return image_dataset

if __name__ == "__main__":
    train_loader = data_utils.DataLoader(
        Xray_Dataset('train_dataset//', train=True),
        batch_size=100,
        shuffle=True, num_workers=2)

    test_loader = data_utils.DataLoader(
        Xray_Dataset('test_dataset/', train=False),
        batch_size=100,
        shuffle=False, num_workers=2)

    for i, (data, label) in enumerate(train_loader):
        print(data.size())
        plt.imshow(data[0].squeeze().numpy(), cmap='gray')
        plt.show()
        print(label[0])
        break