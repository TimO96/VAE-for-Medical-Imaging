# -*- coding: utf-8 -*-
"""
Created on Wed May  8 13:06:11 2019

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

from model import VAE
from conv_layer_architecture import conv_layer_architecture
from distributions import distributions
distribution = distributions()

parser = argparse.ArgumentParser(description='VAE')
parser.add_argument('--architecture-type', type=int, default=3, metavar='T',
                                help='set 1 for lin only, 2 for conv-lin-lin-conv, 3 for conv only')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                                help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                                help='number of epochs to train (default: 10)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                                help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                                help='how many batches to wait before logging training status')
parser.add_argument('--learning-rate', type=int, default=1e-3, metavar='L',
                                help='Learning rate of the optimizer')
parser.add_argument('--x-dim', type=int, default=784, metavar='X',
                                help='When a linear layer is used set this to the size of the input')
parser.add_argument('--z-dim', type=int, default=10, metavar='Z',
                                help='When a linear layer is used set this to the size of the input')
parser.add_argument('--image-size', type=int, default=28, metavar='I',
                                help='resizes the images of the dataset to the set size')

parser.add_argument('--beta', type=int, default=[0, 1, 0], metavar='B',
                                help='set the beta parameter for b-VAE from 0 to 1 in the first 90% if [0, 1, .9]')
parser.add_argument('--final-beta', type=int, default=5, metavar='B',
                                help='set the beta parameter for the last epoch')

parser.add_argument('--base-distribution', type=int, default=distribution.normal, metavar='T',
                                help='set the beta parameter for the last epoch')
parser.add_argument('--loss-distribution', type=int, default=distribution.bernoulli_loss, metavar='T',
                                help='set the beta parameter for the last epoch')
parser.add_argument('--target-distribution', type=int, default=distribution.normal, metavar='T',
                                help='set the beta parameter for the last epoch')

args = parser.parse_args()

# activation for the parameters of the model incase of normal dist mu and sigma and for the loss function
conv_encoder_layers = [
    nn.Conv2d(1, 32, kernel_size=4, stride=1, padding=2),
    nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=2),
    nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=2),
    nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=2),
    nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
    #nn.ReLU(True),
    #nn.MaxPool2d(2, 2),
    #nn.Conv2d(20, 49, 5, 1),
    #nn.ReLU(True),
    #nn.MaxPool2d(2, 2)
    ]

en_conv_1 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=0)
en_conv_2 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=0)
    
encoder_layers = [
    nn.Linear(args.x_dim, 500),
    nn.ReLU(True),
    nn.Linear(500, 200),
    nn.ReLU(True),
    nn.Linear(200, 64),
    nn.ReLU(True),
    nn.Linear(64, args.z_dim*2)
    ]

decoder_layers = [
    nn.Linear(args.z_dim, 64),
    nn.ReLU(True),
    nn.Linear(64, 200),
    nn.ReLU(True),
    nn.Linear(200, 500),
    nn.ReLU(True),
    nn.Linear(500, args.x_dim)
    ]

conv_decoder_layers = [
nn.ConvTranspose2d(64, 64, kernel_size=4, stride=1, padding=1, output_padding=0),
    nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=2, output_padding=0),
    nn.ConvTranspose2d(64, 64, kernel_size=4, stride=1, padding=2, output_padding=0),
    nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=2, output_padding=0),
    #nn.ReLU(True),
    nn.ConvTranspose2d(32, 32, kernel_size=4, stride=1, padding=1, output_padding=0),
    nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=0, output_padding=0)
    ]
activation_layer = [[], [nn.Softplus()], [nn.Sigmoid()]]


def load_data():
    train_data = datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size, shuffle=True, **{})

    test_data = datasets.MNIST('../data', train=False,
                       transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=args.batch_size, shuffle=True, **{})
    return train_data, train_loader, test_data, test_loader

def layer_order(choice):
    options = [[None, encoder_layers, decoder_layers, None, None, None],
                [conv_encoder_layers, encoder_layers, decoder_layers, conv_decoder_layers, None, None],
                [conv_encoder_layers, None, None, conv_decoder_layers, en_conv_1, en_conv_2]]
    if args.architecture_type == choice:
        args.z_dim =get_z_dim(conv_encoder_layers, en_conv_1)
    return options[choice - 1]

def calc_output(w, k, p, s):
    return int(((w - k + 2 * p) / s) + 1)

def calc_padding(o, w, k, s):
    return int((o * s - s - w + k) / 2)

def calc_stride(o, w, k, p):
    return (w - k + 2 * p) / (o - 1)
                
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data
        optimizer.zero_grad()
        x_hat, loss = model(data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

def get_z_dim(encoder, loc_encoder):
    for batch_idx, (data, _) in enumerate(train_loader):
        channels_in = data.shape[1]
        break
    test_input = torch.zeros((1, channels_in, args.image_size,args.image_size))
    conv_encoder = nn.Sequential(*encoder)
    en_conv_1 = nn.Sequential(loc_encoder)
    
    out = conv_encoder(test_input)
    loc =  en_conv_1(out)
    return loc.shape[1] * loc.shape[2] * loc.shape[3]

###############################################################################
if __name__ == "__main__":
    kwargs = {'num_workers': 8, 'pin_memory': False}
    
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
       
#    --------------------------------------------------------------------------
#    train_loader = data_utils.DataLoader(
#            SkinCancerData('./dataset/', augmentation=False),
#            batch_size=args.batch_size,
#            shuffle=True, **kwargs)
#    --------------------------------------------------------------------------
    train_data, train_loader, test_data, test_loader = load_data()


#    --------------------------------------------------------------------------   
    # set 1 for lin only, 2 for conv-lin-lin-conv, 3 for conv only
    layers = layer_order(args.architecture_type)
       
###############################################################################
# p_z still dependent on the parameters per distribution used
    p_z = args.base_distribution(loc=torch.zeros(1,args.z_dim), scale=1) #p_z = Beta(torch.tensor([0.3, 0.3]), torch.tensor([0.3, 0.3]))
    
    # target distribution
    q_z = args.target_distribution
    
    # loss function
    loss_dist = args.loss_distribution
    
    model = VAE(layers, activation_layer, p_z, q_z, loss_dist)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    betas = np.ones(args.epochs)
    if args.beta[2] > 0 and args.beta[2] <= 1:
        adeptive_betas = np.arange(args.beta[0], args.beta[1], args.beta[1] / round(args.beta[2]*args.epochs))
        betas[:len(adeptive_betas)] = adeptive_betas     
    betas[-1] = args.final_beta

    for epoch in range(1, args.epochs + 1):
        model.set_beta(betas[epoch - 1])
        train(epoch)
            #if epoch <= 10:
            #    model.set_beta(betas[epoch-1])
            #with torch.no_grad():
            #    sample = torch.randn(64, 20)
            #    sample = model.decode(sample)
            #    save_image(sample.view(64, 1, 28, 28),
            #               'results/sample_' + str(epoch) + '.png')
    
    print("done")
        