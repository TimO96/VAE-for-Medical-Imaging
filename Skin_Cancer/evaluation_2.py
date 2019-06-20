# -*- coding: utf-8 -*-
"""
Created on Fri May 24 13:20:55 2019

@author: Joris
"""

from __future__ import print_function
import argparse
import torch
import sys
import os
import math
import torch.utils.data
import pandas as pd
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.distributions import Normal, Laplace, Independent, Bernoulli, Gamma, Uniform, Beta
from torch.distributions.kl import kl_divergence
from sklearn.manifold import TSNE
import pickle

import model_VAE
import data_loader
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, DBSCAN
 

        
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(torch.device("cpu"))
#    model.double()
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    print('train loss', checkpoint['train loss'])
    print('test loss', checkpoint['test loss'])

    return model, checkpoint

def split_labels(z, y):
    label = {0 : [[],[]],
              1 : [[],[]],
              2 : [[],[]],
              3 : [[],[]],
              4 : [[],[]],
              5 : [[],[]],
              6 : [[],[]]} 
    for i in range(len(z)):
        label[y[i]][0].append(z[i])
        label[y[i]][1].append(y[i])
    return label

def cluster_eval(x, y, clusters, labels):
    tsne = TSNE(n_components=2, random_state=0)
    X_2d = tsne.fit_transform(x)
    
    target_ids = range(len(y))
    
    plt.figure(figsize=(6, 5))
    colors = 'r', 'g', 'b', 'y', 'gray', 'orange', 'purple'
    for i, c, label in zip(target_ids, colors, labels):
        plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c, label=label)
    plt.legend()
    plt.show()
#    shows T-SNE plot of linear/ convolutional trained model
    cluster = AgglomerativeClustering(n_clusters=clusters, affinity='euclidean', linkage='ward')  
    cluster.fit_predict(x)  
    count = 0 
    for i in range(len(ys)):
        if cluster.labels_[i] == ys[i]:
            count += 1
    print(count/len(ys))  
            
    


if __name__ == "__main__":
    model, checkpoint = load_checkpoint('models/model_fcn_mixture_gaussian_100.pth')
    plt.plot(checkpoint['test loss'])
    plt.show()
    print("test loss")
    train_loss = np.array(checkpoint['train loss'])[:,0]
    plt.plot(train_loss)
    plt.show()
    print("train loss")
    train_loss = np.array(checkpoint['train loss'])[:,1]
    plt.plot(train_loss)
    plt.show()
    print("mixture")
    train_loss = np.array(checkpoint['train loss'])[:,2]
    plt.plot(train_loss)
    plt.show()
    print("KLD")
#    x_hats = checkpoint['test z']
#    ys = checkpoint['test label']
#    losses = checkpoint['loss']
    
#    print(testmodel)
#    
    kwargs = {'num_workers': 8, 'pin_memory': False}
    path_test = 'train_dataset/'
    test_loader = torch.utils.data.DataLoader(
    data_loader.Xray_Dataset(path_test),
    batch_size=64, shuffle=True, **kwargs)

    print("Data is loaded")
    model.eval()
    count = 0
    with torch.no_grad():
        print("starting")
        for batch_idx, (data, label) in enumerate(test_loader):
            z = model.get_z(data)

            if count == 0:
                ys = label.numpy()
                zs = z.numpy()
            else: 
                ys = np.append(ys, label.numpy())
                zs = np.append(zs, z.numpy(), axis=0)
            count += 1
            
    print(np.shape(zs))
    print(np.shape(ys))
    zs = zs.reshape(2378, 64) # 1001 or 2378
    cluster_eval(zs, ys, 7, [0, 1, 2, 3, 4, 5, 6])
    cluster_labels = split_labels(zs, ys)
    for i in range(6):
        for j in range(i+1,7):
            a = np.array(cluster_labels[i][0])
            b = np.array(cluster_labels[j][0])
            zs = np.append(a,b,axis=0)
            a = np.zeros(len(cluster_labels[i][1]))
            b = np.ones(len(cluster_labels[j][1]))
            ys = np.append(a,b,axis=0)
            cluster_eval(zs, ys, 2,[cluster_labels[i][1][0], cluster_labels[j][1][0]])
    
    