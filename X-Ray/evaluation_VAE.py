from __future__ import print_function
import argparse
import torch
import torch.utils.data
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.distributions import Normal, Laplace, Independent, Bernoulli, Gamma, Uniform, Beta
from torch.distributions.kl import kl_divergence

from sklearn.manifold import TSNE
import data_loader

def tsne_plot(model, check_digits):
    path = 'chest-xray-pneumonia/chest_xray/'

    train_loader_plot = torch.utils.data.DataLoader(
        data_loader.Xray_Dataset(path, train=True),
        batch_size=1, shuffle=True, **{})

    z_dims = [[] for i in range(10)]
    x_dims = [[] for i in range(10)]

    for batch_idx, (data, label) in enumerate(train_loader_plot):
        if label.item() in check_digits:
            x_hat, _, _, z = model(data)
            z_dims[label.item()].append(z.detach().numpy()[0])
            x_dims[label.item()].append(x_hat.view(1, -1).detach().numpy()[0])
            if len(z_dims[label.item()]) == 500:
                check_digits.remove(label.item())
            if not check_digits:
                break

    plt.figure(figsize=(10, 10))
    for i in range(len(z_dims)):
        print("nr. " + str(i) + " recons:" + str(len(z_dims[i])))

    for i, data in enumerate(z_dims):
        if data:
            tsne = TSNE(n_components=2, perplexity=50, n_iter=10000, init='pca', random_state=0).fit_transform(data)
            plt.scatter(tsne[:, 0], tsne[:, 1], label=i)

    plt.legend(loc=1)
    plt.show()

    plt.figure(figsize=(10, 10))

    for i, data in enumerate(x_dims):
        if data:
            tsne = TSNE(n_components=2, perplexity=50, n_iter=10000, init='pca', random_state=0).fit_transform(data)
            plt.scatter(tsne[:, 0], tsne[:, 1], label=i)

    plt.legend(loc=1)
    plt.show()
