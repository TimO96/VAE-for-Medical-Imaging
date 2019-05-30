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
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import save_image


from model import VAE
from conv_layer_architecture import conv_layer_architecture
from distributions import distributions
from data_loader import SkinCancerData
distribution = distributions()

parser = argparse.ArgumentParser(description='VAE')
parser.add_argument('--architecture-type', type=int, default=3, metavar='T',
                                help='set 1 for lin only, 2 for conv-lin-lin-conv, 3 for conv only')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                                help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                                help='number of epochs to train (default: 10)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                                help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                                help='how many batches to wait before logging training status')
parser.add_argument('--learning-rate', type=int, default=1e-3, metavar='L',
                                help='Learning rate of the optimizer')
parser.add_argument('--x-dim', type=int, default=4096, metavar='X',
                                help='When a linear layer is used set this to the size of the input')
parser.add_argument('--z-dim', type=int, default=20, metavar='Z',
                                help='When a linear layer is used set this to the size of the input')
parser.add_argument('--image-size', type=int, default=64, metavar='I',
                                help='resizes the images of the dataset to the set size')

parser.add_argument('--beta', type=int, default=[0, 1, 0], metavar='B',
                                help='set the beta parameter for b-VAE from 0 to 1 in the first 90% if [0, 1, .9]')
parser.add_argument('--final-beta', type=int, default=5, metavar='B',
                                help='set the beta parameter for the last epoch')

parser.add_argument('--base-distribution', type=int, default=distribution.normal, metavar='T',
                                help='set the beta parameter for the last epoch')
parser.add_argument('--loss-distribution', type=int, default=distribution.laplace_loss, metavar='T',
                                help='set the beta parameter for the last epoch')
parser.add_argument('--target-distribution', type=int, default=distribution.normal, metavar='T',
                                help='set the beta parameter for the last epoch')

args = parser.parse_args()

# activation for the parameters of the model incase of normal dist mu and sigma and for the loss function
conv_encoder_layers = [
    nn.Conv2d( 3, 10, kernel_size= 5, stride= 1, padding= 2),
    nn.Conv2d(10, 20, kernel_size= 4, stride= 2, padding= 4),
    nn.ReLU(True),
    nn.Conv2d(20, 30, kernel_size= 5, stride= 1, padding= 2),
    nn.Conv2d(30, 40, kernel_size= 4, stride= 2, padding= 4),
    nn.ReLU(True),
    nn.Conv2d(40, 50, kernel_size= 5, stride= 1, padding= 2),
    nn.Conv2d(50, 60, kernel_size= 5, stride= 1, padding= 0),
    nn.ReLU(True),
    nn.Conv2d(60, 70, kernel_size= 5, stride= 1, padding= 2),
    nn.Conv2d(70, 80, kernel_size= 5, stride= 1, padding= 0),
    nn.ReLU(True),
    nn.Conv2d(80, 90, kernel_size= 4, stride= 2, padding= 3),
    nn.Conv2d(90, 100, kernel_size= 5, stride= 1, padding= 2)
    #nn.ReLU(True),
    #nn.MaxPool2d(2, 2),
    #nn.Conv2d(20, 49, 5, 1),
    #nn.ReLU(True),
    #nn.MaxPool2d(2, 2)
    ]

en_conv_1 = nn.Conv2d(100, 100, kernel_size= 4, stride= 2, padding= 4)
en_conv_2 = nn.Conv2d(100, 100, kernel_size= 4, stride= 2, padding= 4)
    
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
      nn.ConvTranspose2d(100, 100, kernel_size= 4, stride= 2, padding= 4, output_padding= 0),
    nn.ConvTranspose2d(100, 100, kernel_size= 5, stride= 1, padding= 2, output_padding= 0),
    nn.ReLU(True),
    nn.ConvTranspose2d(100, 100, kernel_size= 4, stride= 2, padding= 3, output_padding= 0),
    nn.ConvTranspose2d(100, 100, kernel_size= 5, stride= 1, padding= 0, output_padding= 0),
    nn.ReLU(True),
    nn.ConvTranspose2d(100, 100, kernel_size= 5, stride= 1, padding= 2, output_padding= 0),
    nn.ConvTranspose2d(100, 100, kernel_size= 5, stride= 1, padding= 0, output_padding= 0),
    nn.ReLU(True),
    nn.ConvTranspose2d(100, 100, kernel_size= 5, stride= 1, padding= 2, output_padding= 0),
    nn.ConvTranspose2d(100, 100, kernel_size= 4, stride= 2, padding= 4, output_padding= 1),
    nn.ReLU(True),
    nn.ConvTranspose2d(100, 100, kernel_size= 5, stride= 1, padding= 2, output_padding= 0),
    nn.ConvTranspose2d(100, 100, kernel_size= 4, stride= 2, padding= 4, output_padding= 0),
    nn.ConvTranspose2d(100,  100, kernel_size= 5, stride= 1, padding= 2, output_padding= 0),
    ]
activation_layer = [[], [nn.Softplus()], [nn.Softplus()]]


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
    print(1)  
    options = [[None, encoder_layers, decoder_layers, None, None, None],
                [conv_encoder_layers, encoder_layers, decoder_layers, conv_decoder_layers, None, None],
                [conv_encoder_layers, None, None, conv_decoder_layers, en_conv_1, en_conv_2]]
    print(1)  
    if args.architecture_type == 3:
        args.z_dim = get_z_dim(conv_encoder_layers, en_conv_1)
    print(1)  
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
    loss = train_loss / len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, loss))
    return loss
    
def test(epoch):
#    model.eval()
#    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(train_loader):
            save_reconstructions(model, data)
            if i == 2:
                break
#            data = data
#            x_hat, loss = model(data)
#            test_loss += loss.item()
#            if i == 0:
#                n = min(data.size(0), 8)
#                comparison = torch.cat([data[:n],
#                                      x_hat.view(args.batch_size, 1, args.image_size, args.image_size)[:n]])
#                save_image(comparison.cpu(),
#                         'results/reconstruction3_' + str(epoch) + '.png', nrow=n)
# 
#    test_loss /= len(train_loader.dataset)
#    print('====> Test set loss: {:.4f}'.format(test_loss))
    
def sample(x_gen):
    n_comps = 10
    logits = x_gen[:, 0:n_comps, :, :]
    sel = torch.argmax(logits,  # -
                       # torch.log(- torch.log(self.float_tensor(logits.size()).uniform_(1e-5, 1-1e-5))),
                       dim=1, keepdim=True)
    one_hot = torch.zeros(logits.size())
    if torch.cuda.is_available():
        one_hot = one_hot.cuda()
    one_hot.scatter_(1, sel, 1.0)

    # log_scale_r = torch.sum(torch.clamp(x_gen[:, 7 * n_comps:8 * n_comps, :, :] +
    #                                     self.decoder.x_var[0, 0, 0, 0], min=-7.) * one_hot, 1, keepdim=True)
    # log_scale_g = torch.sum(torch.clamp(x_gen[:, 8 * n_comps:9 * n_comps, :, :] +
    #                                     self.decoder.x_var[0, 1, 0, 0], min=-7.) * one_hot, 1, keepdim=True)
    # log_scale_b = torch.sum(torch.clamp(x_gen[:, 9 * n_comps:10 * n_comps, :, :] +
    #                                     self.decoder.x_var[0, 2, 0, 0], min=-7.) * one_hot, 1, keepdim=True)

    mean_x_r = torch.sum(x_gen[:, n_comps:2 * n_comps, :, :] * one_hot, 1, keepdim=True)
    # u_r = self.float_tensor(mean_x_r.size()).uniform_(1e-5, 1 - 1e-5)
    x_r = F.hardtanh(mean_x_r,  # + torch.exp(log_scale_r) * (torch.log(u_r) - torch.log(1. - u_r)),
                     min_val=0., max_val=1.)

    mean_x_g = torch.sum(x_gen[:, 2 * n_comps:3 * n_comps, :, :] * one_hot, 1, keepdim=True) + \
               torch.tanh(torch.sum(x_gen[:, 3 * n_comps:4 * n_comps] * one_hot, 1, keepdim=True)) * x_r
    # u_g = self.float_tensor(mean_x_g.size()).uniform_(1e-5, 1 - 1e-5)
    x_g = F.hardtanh(mean_x_g,  # + torch.exp(log_scale_g) * (torch.log(u_g) - torch.log(1. - u_g)),
                     min_val=0., max_val=1.)

    mean_x_b = torch.sum(x_gen[:, 4 * n_comps:5 * n_comps, :, :] * one_hot, 1, keepdim=True) + \
               torch.tanh(torch.sum(x_gen[:, 5 * n_comps:6 * n_comps] * one_hot, 1, keepdim=True)) * x_r + \
               torch.tanh(
                   torch.sum(x_gen[:, 6 * n_comps:7 * n_comps, :, :] * one_hot, 1, keepdim=True)) * x_g
    # u_b = self.float_tensor(mean_x_b.size()).uniform_(1e-5, 1 - 1e-5)
    x_b = F.hardtanh(mean_x_b,  # + torch.exp(log_scale_b) * (torch.log(u_b) - torch.log(1. - u_b)),
                     min_val=0., max_val=1.)

    sample = torch.cat([x_r, x_g, x_b], 1)
    return sample

def save_reconstructions(model, x):
    # Save reconstuction
    with torch.no_grad():
        x_recon, _ = model.forward(x)
        x_recon = x_recon.view(32,100,64,64)

        sample_t = sample(x_recon[:8])

        n = min(x.size(0), 8)
        comparison = torch.cat([x[:n],
                                sample_t])
        save_image(comparison.cpu(),
                   'reconstruction_diva_resnet_top10_beta_' + str(epoch) + '.png', nrow=n)


def get_z_dim(encoder, loc_encoder):
    print(train_loader)
    for batch_idx, (data, _) in enumerate(train_loader):
        print(data.shape[1])
        channels_in = data.shape[1]
        break
    print(11)
    test_input = torch.zeros((1, channels_in, args.image_size,args.image_size))
    print(11)
    conv_encoder = nn.Sequential(*encoder)
    print(11)
    en_conv_1 = nn.Sequential(loc_encoder)
    print(11)
    out = conv_encoder(test_input)
    print(11)
    loc =  en_conv_1(out)
    return loc.shape[1] * loc.shape[2] * loc.shape[3]

###############################################################################
if __name__ == "__main__":
    kwargs = {'num_workers': 8, 'pin_memory': False}
    
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
       
#    --------------------------------------------------------------------------
    train_loader = data_utils.DataLoader(
            SkinCancerData('./dataset/', augmentation=False, size=args.image_size, train=True, n=1),
            batch_size=args.batch_size,
            shuffle=True, **kwargs)
    print("Data is loaded")
    for batch_idx, (data, label) in enumerate(train_loader):
        print(data.shape, label.shape)
        break
#    --------------------------------------------------------------------------
#    train_data, train_loader, test_data, test_loader = load_data()


#    --------------------------------------------------------------------------   
    # set 1 for lin only, 2 for conv-lin-lin-conv, 3 for conv only
    layers = layer_order(args.architecture_type)
    print(1)   
# p_z still dependent on the parameters per distribution used
    p_z = args.base_distribution(loc=torch.zeros(1,args.z_dim), scale=1) #p_z = Beta(torch.tensor([0.3, 0.3]), torch.tensor([0.3, 0.3]))
    print(2)
    # target distribution
    q_z = args.target_distribution
    print(3)
    # loss function
    loss_dist = args.loss_distribution
    print(4)
    model = VAE(layers, activation_layer, p_z, q_z, loss_dist, args.image_size)
    print(5)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    print(6)
    betas = np.ones(args.epochs)
    if args.beta[2] > 0 and args.beta[2] <= 1:
        adeptive_betas = np.arange(args.beta[0], args.beta[1], args.beta[1] / round(args.beta[2]*args.epochs))
        betas[:len(adeptive_betas)] = adeptive_betas     
    betas[-1] = args.final_beta
    print("Model is set")
    print("Start training")
###############################################################################
    for epoch in range(1, args.epochs + 1):
        model.set_beta(betas[epoch - 1])
        loss = train(epoch)
        test(epoch)
            #if epoch <= 10:
            #    model.set_beta(betas[epoch-1])
            #with torch.no_grad():
            #    sample = torch.randn(64, 20)
            #    sample = model.decode(sample)
            #    save_image(sample.view(64, 1, 28, 28),
            #               'results/sample_' + str(epoch) + '.png')
    
    print("done")
    checkpoint = {'model': model,
                  'state_dict': model.state_dict(),
                  'optimizer' : optimizer.state_dict(),
                  'epochs' : args.epochs,
                  'learning rate' : args.learning_rate,
                  'image size' : args.image_size,
                  'batch size' : args.batch_size,
                  'base dsitribution' : args.base_distribution,
                  'loss dsitribution' : args.loss_distribution,
                  'target dsitribution' : args.target_distribution,
                  'beta' : betas,
                  'loss' : loss
                  }

    torch.save(checkpoint, "models/test.pth")
        