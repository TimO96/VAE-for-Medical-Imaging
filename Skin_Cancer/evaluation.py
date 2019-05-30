# -*- coding: utf-8 -*-
"""
Created on Fri May 24 13:20:55 2019

@author: Joris
"""

import numpy as np
import torch
import torch.utils.data as data_utils
import torchvision.transforms as transforms
import torchvision.transforms.functional
import argparse
import torch
import numpy as np
import torch.utils.data
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from data_loader import SkinCancerData
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from model import VAE

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    print('epochs: ', checkpoint['epochs'])   
    print('learning rate: ', checkpoint['learning rate'])   
    print('image size: ', checkpoint['image size'])   
    print('batch size: ', checkpoint['batch size'])   
    print('betas: ', checkpoint['beta'])  
    print('loss: ', checkpoint['loss'])  
    print('base dsitribution: ', checkpoint['base dsitribution'])   
    print('loss dsitribution: ', checkpoint['loss dsitribution'])   
    print('target dsitribution: ', checkpoint['target dsitribution'])          
    return model, checkpoint['image size'], checkpoint['batch size']



if __name__ == "__main__":
    testmodel, img_size, batch_size = load_checkpoint('models/test.pth')
    print(testmodel)
    kwargs = {'num_workers': 8, 'pin_memory': False}
    train_loader = data_utils.DataLoader(
                SkinCancerData('./dataset/', augmentation=False, size=img_size),
                batch_size=batch_size,
                shuffle=True, **kwargs)
    print("Data is loaded")
    count = 0
    with torch.no_grad():
        print("starting")
        for batch_idx, (data, label) in enumerate(train_loader):
            x_hat = testmodel.get_z(data)

            if count == 0:
                ys = label.numpy()
                x_hats = x_hat.numpy()
            else: 
                ys = np.append(ys, label.numpy())
                x_hats = np.append(x_hats, x_hat.numpy(), axis=0)
#            if count >= 16:
#                 break
            count += 1
            
    print(np.shape(x_hats))
    print(np.shape(ys))
            
    tsne = TSNE(n_components=2, random_state=0)
    X_2d = tsne.fit_transform(x_hats)
    
    target_ids = range(len(ys))
    
    plt.figure(figsize=(6, 5))
    colors = 'r', 'g', 'b', 'y', 'gray', 'orange', 'purple'
    for i, c, label in zip(target_ids, colors, [0, 1, 2, 3, 4, 5, 6]):
        plt.scatter(X_2d[ys == i, 0], X_2d[ys == i, 1], c=c, label=label)
    plt.legend()
    plt.show()
    # shows T-SNE plot of linear/ convolutional trained model