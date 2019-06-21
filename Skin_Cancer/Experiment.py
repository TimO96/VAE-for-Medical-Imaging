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
import pickle

import model_VAE
import data_loader


parser = argparse.ArgumentParser(description='VAE XRAY')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--arch', default='fcn', metavar='N',
                    help='architecture')
parser.add_argument('--distribution', default='laplace', metavar='N',
                    help='architecture')
parser.add_argument('--loss', default='CE', metavar='N',
                    help='architecture')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--tsne', action='store_true',
                    help='t-sne plot added')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(args.seed)
torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)
kwargs = {'num_workers': 8, 'pin_memory': False}

path_train = 'train_dataset/'
path_test = 'test_dataset/'


train_loader = torch.utils.data.DataLoader(
    data_loader.Xray_Dataset(path_train),
    batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    data_loader.Xray_Dataset(path_test),
    batch_size=args.batch_size, shuffle=True, **kwargs)

'''
train_data = datasets.MNIST('../data', train=True, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size, shuffle=True, **kwargs)
test_data = datasets.MNIST('../data', train=False, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=args.batch_size, shuffle=True, **kwargs)
'''

x_dim = 64
z_dim = 64
beta_final=1
arch = args.arch
if args.distribution == 'gaussian':
    distri = Normal
else:
    distri = Laplace

model = model_VAE.VAE(x_dim**2, z_dim, arch, args.distribution, args.loss).to(device)
#filename = 'finalized_model_retina_normalnormal.sav'
#model = pickle.load(open(filename, 'rb'))

optimizer = optim.Adam(model.parameters(), lr=1e-3)

loss = nn.CrossEntropyLoss()

def loss_function(x_hat, x, q_z, z, epoch):
    if args.loss=='mixture':
        BCE = torch.mean(-log_mix_dep_Logistic_256(x, x_hat, average=True, n_comps=10))

    if args.loss=='CE':
        x_hat = x_hat.view(-1, 3, 256, 64, 64)
        x_hat = x_hat.permute(0, 1, 3, 4, 2)
        x_hat = x_hat.contiguous()
        x_hat = x_hat.view(-1, 256)
        #x_hat = torch.round(256 * x_hat.view(-1, 256))
        target = Variable(x.data.view(-1) * 255).long()
        BCE = loss(x_hat, target)
    #x = x.view(-1, x_hat.size(1))
    #tensor = torch.ones(1)
    #p_x_dist = Beta(tensor.new_full((z.size(0), z_dim), 0.5).to(device), tensor.new_full((z.size(0), z_dim), 0.5).to(device))
    z_sqrt = int(np.sqrt(z_dim))
    if arch == 'convlin':
        p_x_dist = Normal(torch.zeros(z.size(0), z_dim).to(device), torch.ones(z.size(0), z_dim).to(device))
    else:
        p_x_dist = Normal(torch.zeros(z.size(0), 1, z_sqrt, z_sqrt).to(device), torch.ones(z.size(0), 1, z_sqrt, z_sqrt).to(device))
    one_third = round(args.epochs/3)

    if beta_final>=1:
        if epoch<=one_third:
            beta = (beta_final*epoch)/one_third
        else:
            beta = beta_final
    else:
        beta = 1

    #BCE = torch.sum(-p_x.log_prob(x.view(x.size(0), x_dim**2)))
    KLD = torch.mean(q_z.log_prob(z) - p_x_dist.log_prob(z))

    print(BCE, KLD, beta)

    return (BCE + beta*KLD), BCE, KLD

def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    # TF ordering
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))

def log_mix_dep_Logistic_256(x, params, average=False, n_comps=10):

    bin_size = 1. / 255.
    logits = params[:, 0:n_comps, :, :]
    means_r = params[:, n_comps:2 * n_comps, :, :]
    means_g = params[:, 2 * n_comps:3 * n_comps, :, :] + torch.tanh(params[:, 3 * n_comps:4 * n_comps]) * x[:, 0:1,
                                                                                                          :, :]
    means_b = params[:, 4 * n_comps:5 * n_comps, :, :] + torch.tanh(params[:, 5 * n_comps:6 * n_comps]) * x[:, 0:1,
                                                                                                          :, :] + \
              torch.tanh(params[:, 6 * n_comps:7 * n_comps, :, :]) * x[:, 1:2, :, :]

    log_scale_r = torch.clamp(params[:, 7 * n_comps:8 * n_comps, :, :], min=-7.)
    log_scale_g = torch.clamp(params[:, 8 * n_comps:9 * n_comps, :, :], min=-7.)
    log_scale_b = torch.clamp(params[:, 9 * n_comps:10 * n_comps, :, :], min=-7.)

    # final size is [B, N_comps, H, W, C]
    means = torch.cat([means_r[:, :, :, :, None], means_g[:, :, :, :, None], means_b[:, :, :, :, None]], 4)
    logvars = torch.cat(
        [log_scale_r[:, :, :, :, None], log_scale_g[:, :, :, :, None], log_scale_b[:, :, :, :, None]], 4)
    # final size is [B, C, H, W, N_comps]
    means = means.transpose(4, 1)
    logvars = logvars.transpose(4, 1)
    x = x[:, :, :, :, None]

    # calculate log probs per component
    # inv_scale = torch.exp(- logvar)[:, :, :, :, None]
    inv_scale = torch.exp(- logvars)
    centered_x = x - means
    inp_cdf_plus = inv_scale * (centered_x + .5 * bin_size)
    inp_cdf_minus = inv_scale * (centered_x - .5 * bin_size)
    cdf_plus = torch.sigmoid(inp_cdf_plus)
    cdf_minus = torch.sigmoid(inp_cdf_minus)

    # bin for 0 pixel is from -infinity to x + 0.5 * bin_size
    log_cdf_zero = F.logsigmoid(inp_cdf_plus)  # cdf_plus
    # bin for 255 pixel is from x - 0.5 * bin_size till infinity
    log_cdf_one = F.logsigmoid(- inp_cdf_minus)  # 1. - cdf_minus

    # calculate final log-likelihood for an image
    mask_zero = (x.data == 0.).float()
    mask_one = (x.data == 1.).float()

    log_logist_256 = mask_zero * log_cdf_zero + (1 - mask_zero) * mask_one * log_cdf_one + \
                     (1 - mask_zero) * (1 - mask_one) * torch.log(cdf_plus - cdf_minus + 1e-7)
    # [B, H, W, n_comps]
    log_logist_256 = torch.sum(log_logist_256, 1) + F.log_softmax(logits.permute(0, 2, 3, 1), 3)

    # log_sum_exp for n_comps
    #log_logist_256 = log_sum_exp(log_logist_256)

    # flatten to [B, H * W]
    log_logist_256 = log_logist_256.view(log_logist_256.size(0), -1)

    # if reduce:
    if average:
        return torch.mean(log_logist_256, 1)
    else:
        return torch.sum(log_logist_256)
        # else:
    #     return log_logist_256


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

def sample_CE(x_gen):
    """Sampling Images"""
    sample = torch.zeros(x_gen.size(0), 3, x_dim, x_dim).cuda()
    x_gen = x_gen.view(x_gen.size(0), 3, 256, x_dim, x_dim)
    x_gen = x_gen.permute(0, 1, 3, 4, 2)

    for i in range(x_dim):
        for j in range(x_dim):

            # [batch_size, channel, height, width, 256]

            # out[:, :, i, j]
            # => [batch_size, channel, 256]
            probs = F.softmax(x_gen[:, :, i, j], dim=2).data

            # Sample single pixel (each channel independently)
            for k in range(3):
                # 0 ~ 255 => 0 ~ 1
                pixel = (torch.multinomial(probs[:, k], 1).float() / 255.).reshape(-1)
                sample[:, k, i, j] = pixel

    return sample

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, q_z, z = model(data)
        loss, BCE, KLD = loss_function(recon_batch, data, q_z, z, epoch)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

        del data
        torch.cuda.empty_cache()


    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    return [train_loss / len(train_loader.dataset), BCE, KLD]


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, q_z, z = model(data)
            test_loss += loss_function(recon_batch, data, q_z, z, epoch)[0].item()
            if i == 0:
                if args.loss=='mixture':
                    sample_t = sample(recon_batch[:8])
                if args.loss=='CE':
                    sample_t = sample_CE(recon_batch[:8])

                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      sample_t])
                print(label[:n])
                if arch == 'resnet':
                    save_image(comparison.cpu(),
                             'reconstruction_resnet_' + args.distribution + '_' + str(epoch) + '.png', nrow=n)
                if arch == 'convlin':
                    save_image(comparison.cpu(),
                             'reconstruction_convlin_'  + args.distribution + '_' + str(epoch) + '.png', nrow=n)
                if arch == 'fcn':
                    save_image(comparison.cpu(),
                             'reconstruction_fcn_'  + args.distribution + '_' + str(epoch) + '.png', nrow=n)
                if arch == 'fcn_resn':
                    save_image(comparison.cpu(),
                             'reconstruction_fcn_resn_'  + args.distribution + '_' + str(epoch) + '.png', nrow=n)

            del data
            torch.cuda.empty_cache()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss


if __name__ == "__main__":
    train_loss = []
    test_loss = []
    for epoch in range(1, args.epochs + 1):
        train_loss.append(train(epoch))
        test_loss.append(test(epoch))

    checkpoint = {'model': model,
              'state_dict': model.state_dict(),
              'optimizer' : optimizer.state_dict(),
              'train loss' : train_loss,
              'test loss' : test_loss
              }
    name = 'models/model_' + args.arch + '_' + args.loss + '_' + args.distribution + '_' + str(epoch) + '.pth'
    torch.save(checkpoint, name)
    #if args.tsne:
#evaluation_VAE.tsne_plot(model, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])