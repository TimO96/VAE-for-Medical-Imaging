import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob

import random
import numpy.random

import scipy.io
from PIL import Image
import PIL

import torch
import torch.utils.data as data_utils
import torchvision.transforms as transforms
import torchvision.transforms.functional

import imgaug.augmenters as iaa


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
    def __init__(self, path, augmentation=False, transform=None, size=64, train=True, n=1):
        self.path = path
        self.augmentation = augmentation
        self.transform = transform
        self.n = n
        self.train = train

        self.to_tensor = transforms.ToTensor()
        self.center_crop = transforms.CenterCrop(300)
        self.resize = transforms.Resize((size, size), interpolation=Image.BILINEAR)
        self.hflip = transforms.RandomHorizontalFlip()
        self.vflip = transforms.RandomVerticalFlip()
        self.rrotate = RandomRotate()
        self.to_pil = transforms.ToPILImage()
        
        self.X, self.Y = self.get_data()

    def read_img_from_file(self, img_name):
        img_path = self.path + img_name + '.jpg'

        with open(img_path, 'rb') as f:
            with Image.open(f) as img:
                img = img.convert('RGB')

        img = self.to_tensor(self.resize(self.center_crop(img)))

        return img

    def train_test_split(self, lesions_dict_training, n):   
        train_data = []
        test_data = []
        train_label = np.array([])
        test_label = np.array([])
        for key in lesions_dict_training.keys():
            train = []
            length = round(len(lesions_dict_training[key])*0.1)
            test_data.append(lesions_dict_training[key][length*(n-1):length*n])
            test_label = np.append(test_label, key * np.ones(length, dtype=int))
            
            train.append(lesions_dict_training[key][length*n:])
            if n > 1:
                train.append(lesions_dict_training[key][0:length*(n-1)])
            train = [j for i in train for j in i]
            if len(lesions_dict_training[key][length:]) >= 463:
                train = train[:463]
            train_data.append(train)
            train_label = np.append(train_label, key * np.ones(len(train), dtype=int))
            
        train_data = [j for i in train_data for j in i]
        test_data = [j for i in test_data for j in i]
        print(len(train_data),len(test_data), train_label.shape,test_label.shape)
        return train_data, test_data, train_label, test_label

    def get_data(self):
        np.random.seed(42)
        tile_df = get_dataframe(self.path)
        lesions_dict_training = {
                                0: [],
                                1: [],
                                2: [],
                                3: [],
                                4: [],
                                5: [],
                                6: []
                                }

        for index, row in tile_df.iterrows():
            #print(index)
            if index % 1000 == 0: # --------------------------------------------------------------- limit data loaded temp
                print("Loaded", index, "of the 10015")
            img = self.read_img_from_file(row['image_id'])
            lesions_dict_training[row['cell_type_idx']].append(img)

        for key in lesions_dict_training.keys():    
            np.random.shuffle(lesions_dict_training[key])
            
        train, test, train_label, test_label = self.train_test_split(lesions_dict_training, self.n)
        if self.train:
            X = torch.stack(train)
            Y = torch.LongTensor(train_label)
        else:
            X = torch.stack(test)
            Y = torch.LongTensor(test_label)
       

        # Convert to onehot
#        y = torch.eye(7)
#        train_labels = y[train_labels]
        print("check")
        return X, Y
    
    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
#        self.training_data, self.testing_data, self.training_labels, self.testing_labels
        x = self.X[index]
        y = self.Y[index]

        if self.augmentation:
            x = self.to_tensor(self.vflip(self.hflip(self.rrotate(self.to_pil(x)))))
        else:
            x = x.clone()

        if self.transform is not None:
            x =  self.to_tensor(self.transform(self.to_pil(x)))

        return x, y


if __name__ == "__main__":
    from torchvision.utils import save_image

    kwargs = {'num_workers': 8, 'pin_memory': False}

    seed = 1
#    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
#    np.random.seed(seed)


    param = np.arange(0, 0.4, 0.05)
    
    transform = torchvision.transforms.Compose([
            torchvision.transforms.ColorJitter(saturation=np.random.choice(param), brightness=np.random.choice(param)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip()
    ])
    for k in range(1,6):
        train_loader = data_utils.DataLoader(
            SkinCancerData('./dataset/', augmentation=False, transform=transform,size=128, train=True, n=k),
            batch_size=10,
            shuffle=False, **kwargs)
#        print("loaded", len(train_loader))
#        for j in range(1):
#    
#            for i, (x, y) in enumerate(train_loader):
#    
#                if i == 0:
#                    save_image(x[:100].cpu(),
#                               'reconstruction_cell_train_' +k+ '.png', nrow=10)

    # print(y_array, d_array)
    
        for batch_idx, (data, label) in enumerate(train_loader):
            print(data.shape, label.shape)
            b = np.zeros((128,128))
    #        b = 0.2989*data[batch_idx,0] + 0.5870*data[batch_idx,1] + 0.1140*data[batch_idx,2]
    #        b[:,:,1] = 0.5870*data[batch_idx,1]
    #        b[:,:,2] = 0.1140*data[batch_idx,2]
            print(data.shape)
            print(label)
            plt.imshow(data[batch_idx,0], cmap='gray')
            plt.show()
            if batch_idx == 2:
                break