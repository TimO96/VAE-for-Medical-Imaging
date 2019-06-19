from __future__ import print_function
import argparse
import torch
import torch.utils.data
import matplotlib.pyplot as plt
import pandas as pd
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.distributions import Normal, Laplace, Independent, Bernoulli, Gamma, Uniform, Beta
from torch.distributions.kl import kl_divergence
import numpy as np
import pickle

from sklearn.manifold import TSNE
import data_loader

if __name__ == "__main__":
    df = pd.read_csv('results_normal-normal_beta1.csv')

    plt.plot(df['Loss'].values, label='total loss', color='r')
    plt.plot(df['Recon Loss'].values, label='Recon Loss', color='b')
    plt.plot(df['KLD'].values, label='KL-Divergence', color='g')
    plt.legend()
    plt.show()

    '''
    path = 'chest-xray-pneumonia/chest_xray/'

    test_loader_plot = torch.utils.data.DataLoader(
        data_loader.Xray_Dataset(path, train=True),
        batch_size=1, shuffle=True, **{})

    filename = 'finalized_model_retina_normalnormal.sav'
    model = pickle.load(open(filename, 'rb'))

    train_data = datasets.MNIST('../data', train=False, transform=transforms.ToTensor())

    test_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, **{})

    count = 0
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            data = data
            x_hat, _, _, _ = model(data)
            if count == 0:
                ys = label.numpy()
                x_hats = x_hat.numpy()
            else:
                ys = np.append(ys, label.numpy())
                x_hats = np.append(x_hats, x_hat.numpy(), axis=0)
            if count >= 8:
                break
            count += 1

    print(np.shape(x_hats))
    print(np.shape(ys))

    tsne = TSNE(n_components=2, random_state=0)
    X_2d = tsne.fit_transform(x_hats.reshape(-1, 784))

    target_ids = range(len(ys))

    plt.figure(figsize=(6, 5))
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'gray', 'orange', 'purple'
    for i, c, label in zip(target_ids, colors, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
        plt.scatter(X_2d[ys == i, 0], X_2d[ys == i, 1], c=c, label=label)
    plt.legend()
    plt.show()

    xv = np.arange(0, 4, .2)
    yv = np.arange(0, 4, .2)
    sample = np.zeros([len(yv)*len(xv), 2])
    counter = 0
    for i, j in zip(xv, yv):
        for j in yv:
            sample[counter] = [i, j]
            counter += 1

    images = model.decode(torch.tensor(sample, dtype=torch.float)).detach().numpy()
    image = np.zeros([len(xv)*28, len(yv)*28])
    counter = 0
    for i in range(len(xv)):
        for j in range(len(yv)):
            image[i*28:i*28+28,j*28:j*28+28] = images[counter].reshape((28,28))
            counter += 1

    plt.figure(figsize=(15, 15))
    plt.imshow(image, cmap='gray')
    plt.show()
    '''
