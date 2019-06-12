# -*- coding: utf-8 -*-
"""
Created on Wed May  8 13:06:11 2019

@author: Joris
"""

import numpy as np
import torch
import torch.nn.functional as F
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
import itertools

from model import VAE
from model_resnet import VAE_RESN
from conv_layer_architecture import conv_layer_architecture
from distributions import distributions
from data_loader import SkinCancerData
distribution = distributions()

parser = argparse.ArgumentParser(description='VAE')
parser.add_argument('--architecture-type', type=int, default=1, metavar='T',
                                help='set 1 for lin only, 2 for conv-lin-lin-conv, 3 for conv only')
parser.add_argument('--loss-type', type=int, default=0, metavar='T',
                                help='1 for CE or 0 for mixture')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                                help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=15, metavar='N',
                                help='number of epochs to train (default: 10)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                                help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                                help='how many batches to wait before logging training status')
parser.add_argument('--learning-rate', type=int, default=1e-3, metavar='L',
                                help='Learning rate of the optimizer')
parser.add_argument('--x-dim', type=int, default=9216, metavar='X',
                                help='When a linear layer is used set this to the size of the input')
parser.add_argument('--z-dim', type=int, default=64, metavar='Z',
                                help='When a linear layer is used set this to the size of the input')
parser.add_argument('--image-size', type=int, default=112, metavar='I',
                                help='resizes the images of the dataset to the set size')

parser.add_argument('--beta', type=int, default=[0, 1, .8], metavar='B',
                                help='set the beta parameter for b-VAE from 0 to 1 in the first 90% if [0, 1, .9]')
parser.add_argument('--final-beta', type=int, default=1, metavar='B',
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
    nn.Conv2d( 3, 12, kernel_size= 4, stride= 2, padding= 3),
    nn.Conv2d(12, 12, kernel_size= 6, stride= 1, padding= 0),
    nn.ReLU(True),
    nn.Conv2d(12, 42, kernel_size= 4, stride= 2, padding= 3),
    nn.Conv2d(42, 60, kernel_size= 6, stride= 1, padding= 0),
    nn.ReLU(True),
    nn.Conv2d(60, 96, kernel_size= 6, stride= 1, padding= 0),
    nn.Conv2d(96, 120, kernel_size= 5, stride= 1, padding= 0),
    nn.ReLU(True),
    nn.Conv2d(120, 144, kernel_size= 5, stride= 1, padding= 1),
    nn.Conv2d(144, 180, kernel_size= 5, stride= 1, padding= 1),
    nn.ReLU(True),
    nn.Conv2d(180, 200, kernel_size= 5, stride= 1, padding= 2),
    nn.Conv2d(200, 220, kernel_size= 5, stride= 1, padding= 1),
    nn.ReLU(True),
    nn.Conv2d(220, 256, kernel_size= 4, stride= 2, padding= 3)
    ]

en_conv_1 = nn.Conv2d(100, 100, kernel_size= 4, stride= 2, padding= 4)
en_conv_2 = nn.Conv2d(100, 100, kernel_size= 4, stride= 2, padding= 4)
    
encoder_layers = [
    nn.Linear(args.x_dim, 8192),
    nn.ReLU(True),
    nn.Linear(8192, 6144),
    nn.ReLU(True),
    nn.Linear(6144, 4096),
    nn.ReLU(True),
    nn.Linear(4096, 3072),
    nn.ReLU(True),
    nn.Linear(3072, 2048),
    nn.ReLU(True),
    nn.Linear(2048, 1536),
    nn.ReLU(True),
    nn.Linear(1536, 768),
    nn.ReLU(True),
    nn.Linear(768, 384),
    nn.ReLU(True),
    nn.Linear(384, 192),
    nn.ReLU(True),
    nn.Linear(192, args.z_dim*2)
    ]

decoder_layers = [
    nn.Linear(args.z_dim, 128),
    nn.ReLU(True),
    nn.Linear(128, 192),
    nn.ReLU(True),
    nn.Linear(192, 384),
    nn.ReLU(True),
    nn.Linear(384, 768),
    nn.ReLU(True),
    nn.Linear(768, 1536),
    nn.ReLU(True),
    nn.Linear(1536, 2048),
    nn.ReLU(True),
    nn.Linear(2048, 3072),
    nn.ReLU(True),
    nn.Linear(3072, 4096),
    nn.ReLU(True),
    nn.Linear(4096, 6144),
    nn.ReLU(True),
    nn.Linear(6144, 8192),
    nn.ReLU(True),
    nn.Linear(8192, args.x_dim)
    ]

conv_decoder_layers_ce = [
    nn.ConvTranspose2d(256, 220, kernel_size= 4, stride= 2, padding= 3, output_padding= 0),
    nn.ReLU(True),
    nn.ConvTranspose2d(220, 200, kernel_size= 5, stride= 1, padding= 1, output_padding= 0),
    nn.ConvTranspose2d(200, 180, kernel_size= 5, stride= 1, padding= 2, output_padding= 0),
    nn.ReLU(True),
    nn.ConvTranspose2d(180, 144, kernel_size= 5, stride= 1, padding= 1, output_padding= 0),
    nn.ConvTranspose2d(144, 120, kernel_size= 5, stride= 1, padding= 1, output_padding= 0),
    nn.ReLU(True),
    nn.ConvTranspose2d(120, 96, kernel_size= 5, stride= 1, padding= 0, output_padding= 0),
    nn.ConvTranspose2d(96, 60, kernel_size= 6, stride= 1, padding= 0, output_padding= 0),
    nn.ReLU(True),
    nn.ConvTranspose2d(60, 42, kernel_size= 6, stride= 1, padding= 0, output_padding= 0),
    nn.ConvTranspose2d(42, 12, kernel_size= 4, stride= 2, padding= 3, output_padding= 1),
    nn.ReLU(True),
    nn.ConvTranspose2d(12, 12, kernel_size= 6, stride= 1, padding= 0, output_padding= 0),
    nn.ConvTranspose2d(12,  100, kernel_size= 4, stride= 2, padding= 3, output_padding= 0)
    ]

conv_decoder_layers_mix = [
    nn.ConvTranspose2d(256, 220, kernel_size= 4, stride= 2, padding= 3, output_padding= 0),
    nn.ReLU(True),
    nn.ConvTranspose2d(220, 200, kernel_size= 5, stride= 1, padding= 1, output_padding= 0),
    nn.ConvTranspose2d(200, 180, kernel_size= 5, stride= 1, padding= 2, output_padding= 0),
    nn.ReLU(True),
    nn.ConvTranspose2d(180, 144, kernel_size= 5, stride= 1, padding= 1, output_padding= 0),
    nn.ConvTranspose2d(144, 120, kernel_size= 5, stride= 1, padding= 1, output_padding= 0),
    nn.ReLU(True),
    nn.ConvTranspose2d(120, 96, kernel_size= 5, stride= 1, padding= 0, output_padding= 0),
    nn.ConvTranspose2d(96, 60, kernel_size= 6, stride= 1, padding= 0, output_padding= 0),
    nn.ReLU(True),
    nn.ConvTranspose2d(60, 42, kernel_size= 6, stride= 1, padding= 0, output_padding= 0),
    nn.ConvTranspose2d(42, 12, kernel_size= 4, stride= 2, padding= 3, output_padding= 1),
    nn.ReLU(True),
    nn.ConvTranspose2d(12, 12, kernel_size= 6, stride= 1, padding= 0, output_padding= 0),
    nn.ConvTranspose2d(12,  3 * 256, kernel_size= 4, stride= 2, padding= 3, output_padding= 0)
    ]
activation_layer = [[], [nn.Softplus()], [nn.Sigmoid()]]
conv_decoder_layers = [conv_decoder_layers_mix, conv_decoder_layers_ce]

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

def layer_order(choice, loss_function):
    options = [[None, encoder_layers, decoder_layers, None, None, None],
                [conv_encoder_layers, encoder_layers, decoder_layers, conv_decoder_layers[loss_function], None, None],
                [conv_encoder_layers, None, None, conv_decoder_layers, en_conv_1, en_conv_2],
                [None, encoder_layers, decoder_layers, loss_function, args.z_dim, args.x_dim]]
    if args.architecture_type == 3:
        args.z_dim = get_z_dim(conv_encoder_layers, en_conv_1)
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
        data = data.to(device)
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
    
def test():
    model.eval()
    test_loss = 0
    X = np.array([])
    Y = np.array([])
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            data = data.to(device)
            X = np.append(X, model.get_z(data).numpy())
            Y = np.append(Y, label.numpy())
            x_hat, loss = model(data)
            test_loss += loss.item()
    test_loss /= len(test_loader.dataset)
    X = X.reshape(int(len(X)/64), 64)
    return test_loss, X, Y
#            sample_image_CE(epoch, model, data)
#            save_reconstructions(model, data)
#            if i == 2:
#                break
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
                   'reconstruction_diva_resnet_top10_beta2_' + str(epoch) + '.png', nrow=n)
        
def sample_image_CE(epoch, model, x):
    """Sampling Images"""

    image_path =  'reconstruction_diva_resnet_top10_beta2_' + str(epoch) + '.png'

    sample = torch.zeros(args.batch_size, 3, 64, 64)
    with torch.no_grad():
        x_recon, _ = model.forward(x)
        x_recon = x_recon.view(args.batch_size, 3, 256, 64, 64)
        x_recon = x_recon.permute(0, 1, 3, 4, 2)    

    sample = torch.zeros(args.batch_size, 3, 64, 64)

    for i in range(64):
        for j in range(64):

            # [batch_size, channel, height, width, 256]

            # out[:, :, i, j]
            # => [batch_size, channel, 256]
            probs = F.softmax(x_recon[:, :, i, j], dim=2).data

            # Sample single pixel (each channel independently)
            for k in range(3):
                # 0 ~ 255 => 0 ~ 1
                pixel = torch.multinomial(probs[:, k], 1).float() / 255.
                sample[:, k, i, j] = pixel


    save_image(sample, image_path)


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

def save_model(loss_function, distri, architecture, epoch, test_loss=None, X=None, Y=None):
    if loss_function == 1:
        loss = 'CE'
    else:
        loss = 'Mixture'
        
    if architecture == 2:
        arch = 'Conv'
    else:
        arch = 'resn'
    print("done")
    checkpoint = {'model': model,
                  'state_dict': model.state_dict(),
                  'optimizer' : optimizer.state_dict(),
                  'epochs' : args.epochs,
                  'learning rate' : args.learning_rate,
                  'image size' : args.image_size,
                  'batch size' : args.batch_size,
                  'base dsitribution' : distri,
                  'loss dsitribution' : arch,
                  'target dsitribution' : distri,
                  'beta' : betas,
                  'loss' : loss,
                  'test loss' : test_loss,
                  'test z' : X,
                  'test label' : Y
                  }
    if distri == distribution.beta:
        dist = 'beta'
    else:
        dist = 'gaussian'
    name = 'models/model_' + str(N) + '_' + arch + '_' + loss + '_' + dist + '_' + str(epoch) + '.pth'
    print(name)
    torch.save(checkpoint, name)

###############################################################################
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    kwargs = {'num_workers': 8, 'pin_memory': False}
    conv_lin = 2
    resn_lin = 4
    CE_loss = 1
    MIX_loss = 0
    conv = VAE
    resn = VAE_RESN
    
    experiments = [[conv, conv_lin, CE_loss,  distribution.normal],
                   [conv, conv_lin, CE_loss,  distribution.beta],
                   [conv, conv_lin, MIX_loss, distribution.normal],
                   [conv, conv_lin, MIX_loss, distribution.beta],
                   [resn, resn_lin, CE_loss,  distribution.normal],
                   [resn, resn_lin, CE_loss,  distribution.beta],
                   [resn, resn_lin, MIX_loss, distribution.normal],
                   [resn, resn_lin, MIX_loss, distribution.beta]
                   ]    
    test_im = torch.ones((128,3,112,112)).to(device)
    for N in range(1, 2):
        train_loader = data_utils.DataLoader(
            SkinCancerData('./dataset/', augmentation=False, size=args.image_size, train=True, n=N),
            batch_size=args.batch_size,
            shuffle=True, **kwargs)
        
        test_loader = data_utils.DataLoader(
            SkinCancerData('./dataset/', augmentation=False, size=args.image_size, train=False, n=N),
            batch_size=args.batch_size,
            shuffle=True, **kwargs)
        for model_arch, architecture, loss_function, distri in experiments:
            #    --------------------------------------------------------------------------

            print("Data is loaded")
            #    --------------------------------------------------------------------------
            #    train_data, train_loader, test_data, test_loader = load_data()
        
        
            #    --------------------------------------------------------------------------   
            # set 1 for lin only, 2 for conv-lin-lin-conv, 3 for conv only
            layers = layer_order(architecture, loss_function)
            # p_z still dependent on the parameters per distribution used
            
            
            if distri == distribution.beta:
                a = (torch.ones(args.z_dim) * 2).to(device)
                b = (torch.ones(args.z_dim) * 2).to(device)
                p_z = distri(a, b)
                activation_layer = [[nn.Softplus()], [nn.Softplus()], [nn.Sigmoid()]]
            else:
                p_z = distri(loc=torch.zeros(1,args.z_dim).to(device), scale=1)
                activation_layer = [[], [nn.Softplus()], [nn.Sigmoid()]]
            
            # target distribution
            q_z = distri
            # loss function
            loss_dist = args.loss_distribution
            model = model_arch(layers, activation_layer, p_z, q_z, args.image_size, loss_function)
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
            model.to(device)
            betas = np.ones(args.epochs)
            if args.beta[2] > 0 and args.beta[2] <= 1:
                adeptive_betas = np.arange(args.beta[0], args.beta[1], args.beta[1] / round(args.beta[2]*args.epochs))
                betas[:len(adeptive_betas)] = adeptive_betas     
            betas[-1] = args.final_beta
            print("Model is set")
            print("Start training")
            loss = []
            model.test_encode_decode(test_im)
#            a = model.get_z(test_im)
#            print(a.shape,a)
        ###############################################################################
            for epoch in range(1, args.epochs + 1):
                model.set_beta(betas[epoch - 1])
                loss.append(train(epoch))
                if epoch % 10 == 0:
                    save_model(loss_function, distri, architecture, epoch)

            test_loss, X, Y = test()
            save_model(loss_function, distri, architecture, epoch, test_loss, X, Y)
        break
                    

            