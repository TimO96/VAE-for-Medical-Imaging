import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob

import random
import numpy.random

import scipy.io
from PIL import Image

import torch
import torch.utils.data as data_utils
import torchvision.transforms as transforms
import torchvision.transforms.functional


def get_dataframe(path):
    base_skin_dir = path
    imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                         for x in glob(os.path.join(base_skin_dir, '*', '*.jpg'))}

    lesion_type_dict = {
        'akiec': 'Actinic keratoses',
        'bcc': 'Basal cell carcinoma',
        'bkl': 'Benign keratosis-like lesions ',
        'df': 'Dermatofibroma',
        'mel': 'dermatofibroma',
        'nv': 'Melanocytic nevi',
        'vasc': 'Vascular lesions',
    }

    tile_df = pd.read_csv(os.path.join(base_skin_dir, 'HAM10000_metadata.csv'))
    tile_df['path'] = tile_df['image_id'].map(imageid_path_dict.get)
    tile_df['cell_type'] = tile_df['dx'].map(lesion_type_dict.get)
    tile_df['cell_type_idx'] = pd.Categorical(tile_df['dx']).codes
    # tile_df['age'] = tile_df['localization'].map(body_part_dict.get)

    tile_df[['cell_type_idx', 'cell_type']].sort_values('cell_type_idx').drop_duplicates()

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


class SkinCancerData(data_utils.Dataset):
    def __init__(self, path, augmentation=False, transform=None, size=64):
        self.path = path
        self.augmentation = augmentation
        self.transform = transform

        self.to_tensor = transforms.ToTensor()
        self.center_crop = transforms.CenterCrop(300)
        self.resize = transforms.Resize((size, size), interpolation=Image.BILINEAR)
        self.hflip = transforms.RandomHorizontalFlip()
        self.vflip = transforms.RandomVerticalFlip()
        self.rrotate = RandomRotate()
        self.to_pil = transforms.ToPILImage()

        self.train_data, self.train_labels = self.get_data()

    def read_img_from_file(self, img_name):
        img_path = self.path + img_name + '.jpg'

        with open(img_path, 'rb') as f:
            with Image.open(f) as img:
                img = img.convert('RGB')

        img = self.to_tensor(self.resize(self.center_crop(img)))

        return img

    def get_data(self):
        tile_df = get_dataframe(self.path)

        imgs_per_domain_list = []
        labels_per_domain_list = []

        for index, row in tile_df.iterrows():
            print(index)
            if index % 100 == 0: # --------------------------------------------------------------- limit data loaded temp
                img = self.read_img_from_file(row['image_id'])
                label = row['cell_type_idx']
    
                imgs_per_domain_list.append(img)
                labels_per_domain_list.append(label)

        # One last cat
        train_imgs = torch.stack(imgs_per_domain_list)
        train_labels = torch.LongTensor(labels_per_domain_list)

        # Convert to onehot
        y = torch.eye(7)
        train_labels = y[train_labels]

        return train_imgs, train_labels

    def __len__(self):
        return len(self.train_labels)

    def __getitem__(self, index):
        x = self.train_data[index]
        y = self.train_labels[index]

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

    age = [7]

    print('Number of domains:', len(age))

    train_loader = data_utils.DataLoader(
        SkinCancerData('./dataset/', augmentation=False),
        batch_size=1,
        shuffle=True, **kwargs)

    for j in range(1):

        for i, (x, y) in enumerate(train_loader):

            if i == 0:
                save_image(x[:100].cpu(),
                           'reconstruction_cell_train_' + '.png', nrow=10)

    # print(y_array, d_array)
    for batch_idx, (data, _) in enumerate(train_loader):
        b = np.zeros((64,64,3))
        b[:,:,0] = data[0,0]
        b[:,:,1] = data[0,1]
        b[:,:,2] = data[0,2]
        plt.imshow(b)
        plt.show()
        if batch_idx == 0:
            break