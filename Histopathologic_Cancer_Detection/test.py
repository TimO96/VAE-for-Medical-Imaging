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


def get_dataframe(path):
    base_skin_dir = path
    imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                         for x in glob(os.path.join(base_skin_dir, '*', '*.tif'))}

    class_dict = {
        '0': 0.0,
        '1': 1.0
    }

    tile_df = pd.read_csv('data/train_labels.csv')
    tile_df['path'] = tile_df['id'].map(imageid_path_dict.get)
    tile_df['class'] = tile_df['label'].map(class_dict.get)
    #tile_df['cell_type_idx'] = pd.Categorical(tile_df['dx']).codes
    # tile_df['age'] = tile_df['localization'].map(body_part_dict.get)
    #tile_df['age_idx'] = pd.Categorical(tile_df['age']).codes

    #tile_df[['cell_type_idx', 'cell_type']].sort_values('cell_type_idx').drop_duplicates()

    return tile_df


class RandomRotate(object):
    """Rotate the given PIL.Image by either 0, 90, 180, 270."""

    def __call__(self, img):
        random_rotation = numpy.random.randint(4, size=1)
        if random_rotation == 0:
            pass
        else:
            img = torchvision.transforms.functional.rotate(img, random_rotation*90)
        return img


class CancerData(data_utils.Dataset):
    def __init__(self, path, augmentation=False, transform=None):
        self.path = path
        self.augmentation = augmentation
        self.transform = transform

        self.to_tensor = transforms.ToTensor()
        self.center_crop = transforms.CenterCrop(300)
        self.resize = transforms.Resize((64, 64), interpolation=Image.BILINEAR)
        self.hflip = transforms.RandomHorizontalFlip()
        self.vflip = transforms.RandomVerticalFlip()
        self.rrotate = RandomRotate()
        self.to_pil = transforms.ToPILImage()

        self.train_data, self.train_labels = self.get_data()

    def read_img_from_file(self, img_name):

        img_path = self.path + img_name + '.tif'

        with open(img_path, 'rb') as f:
            with Image.open(f) as img:
                img = img.convert('RGB')

        img = self.to_tensor(self.resize(self.center_crop(img)))

        return img

    def get_data(self):
        tile_df = get_dataframe(self.path)

        imgs_per_domain_list = []
        labels_per_domain_list = []
        #omain_per_domain_list = []

        for index, row in tile_df.iterrows():
            #print(index)
            print(row['id'])
            print(tile_df['id'])
            if row['id'] in tile_df['id']:
                img = self.read_img_from_file(row['id'])
                label = row['label']
                #domain = row['age_idx']

                imgs_per_domain_list.append(img)
                print(img)
                labels_per_domain_list.append(label)
            #domain_per_domain_list.append(domain)

        # One last cat
        train_imgs = torch.stack(imgs_per_domain_list)
        train_labels = torch.LongTensor(labels_per_domain_list)
        #train_domains = torch.LongTensor(domain_per_domain_list)

        # Convert to onehot
        y = torch.eye(7)
        train_labels = y[train_labels]

        #d = torch.eye(18)
        #train_domains = d[train_domains]

        return train_imgs, train_labels #train_domains

    def __len__(self):
        return len(self.train_labels)

    def __getitem__(self, index):
        x = self.train_data[index]
        y = self.train_labels[index]
        #d = self.train_domain[index]

        if self.augmentation:
            x = self.to_tensor(self.vflip(self.hflip(self.rrotate(self.to_pil(x)))))
        else:
            x = x.clone()

        if self.transform is not None:
            x = self.transform(x)

        return x, y


if __name__ == "__main__":
    from torchvision.utils import save_image

    kwargs = {'num_workers': 8, 'pin_memory': False}

    seed = 1
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


    train_loader = data_utils.DataLoader(
        CancerData('data/train/', augmentation=False),
        batch_size=100,
        shuffle=True, **kwargs)

    for j in range(1):

        y_array = np.zeros(7)
        #d_array = np.zeros(18)

        for i, (x, y) in enumerate(train_loader):

            y_array += y.sum(dim=0).cpu().numpy()
            #d_array += d.sum(dim=0).cpu().numpy()

            if i == 0:
                save_image(x[:100].cpu(),
                           'reconstruction_cell_train_' + str(age[0]) + '.png', nrow=10)

    print('\n')
    for y in y_array:
        print(int(y))
    # print(y_array, d_array)
