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


def Xray_Dataset(path):

    image_dataset = torchvision.datasets.ImageFolder(path,transforms.ToTensor())

    return image_dataset

if __name__ == "__main__":
    train_loader = data_utils.DataLoader(
        Xray_Dataset('chest-xray-pneumonia/chest_xray/train/'),
        batch_size=100,
        shuffle=True, num_workers=2)

    for i, (data, label) in enumerate(train_loader):
        plt.imshow(data, cmap='gray')
        plt.show()
        print(label)
        break
