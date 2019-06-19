from __future__ import print_function
import argparse
import torch
import torch.utils.data
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.distributions import Normal, Laplace, Independent, Bernoulli, Gamma, Uniform, Beta
from torch.distributions.kl import kl_divergence

def normal_dist(mu, var):
    return Normal(loc=mu, scale=var)

def laplace_dist(mu, var):
    return Laplace(loc=mu, scale=var)

def gamma_dist(mu, var):
    return Gamma(concentration=mu, rate=var)

def beta_dist(mu, var):
    return Beta(concentration1=mu, concentration0=var)

def bernoulli_loss(x_hat):
    return Bernoulli(x_hat)

def laplace_loss(x_hat, scale=0.08):
    return Laplace(loc=x_hat, scale=scale)

class VAE(nn.Module):
    def __init__(self, x_dim, z_dim, architecture):
        super(VAE, self).__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.arch = architecture

        self.conv1 = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=4, stride=1, padding=2),
        nn.ELU()
        )

        self.deconv1 = nn.Sequential(
        nn.ConvTranspose2d(100, 100, kernel_size=4, stride=2, padding=1, output_padding=0),
        nn.Upsample(64)
        )

        self.conv_encoder_1 = nn.Sequential(
        nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=2),
        nn.BatchNorm2d(32),
        nn.ELU(),
        nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=2),
        nn.BatchNorm2d(32),
        nn.ELU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=2),
        nn.BatchNorm2d(64),
        nn.ELU(),
        nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=2),
        nn.BatchNorm2d(64),
        nn.ELU()
        )

        self.shortcut_en1 = nn.Sequential(
        nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=2),
        nn.BatchNorm2d(64),
        nn.Upsample(10),
        nn.ELU()
        )

        self.conv_encoder_2 = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=2),
        nn.BatchNorm2d(128),
        nn.ELU(),
        nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=2),
        nn.BatchNorm2d(128),
        nn.ELU()
        )

        self.shortcut_en2 = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(128),
        nn.Upsample(6),
        nn.ELU()
        )

        self.conv_encoder_3 = nn.Sequential(
        nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=2),
        nn.BatchNorm2d(256),
        nn.ELU(),
        nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=0),
        nn.BatchNorm2d(256),
        nn.ELU()
        )

        self.shortcut_en3 = nn.Sequential(
        nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(256),
        nn.Upsample(2),
        nn.ELU()
        )

        self.conv_encoder_4 = nn.Sequential(
        nn.Conv2d(256, 2, kernel_size=4, stride=1, padding=2),
        nn.BatchNorm2d(2),
        nn.ELU(),
        nn.Conv2d(2, 2, kernel_size=4, stride=2, padding=2),
        nn.BatchNorm2d(2),
        nn.Upsample(int(np.sqrt(self.z_dim))),
        nn.Softplus()
        )

        self.linear_encoder = nn.Sequential(
        nn.Linear(256*2*2, 750),
        nn.ReLU(True),
        nn.Linear(750, 392),
        nn.ReLU(True),
        nn.Linear(392, 180),
        nn.ReLU(True),
        nn.Linear(180, self.z_dim*2),
        nn.Softplus()
        )

        self.linear_decoder = nn.Sequential(
        nn.Linear(self.z_dim, 180),
        nn.ReLU(True),
        nn.Linear(180, 392),
        nn.ReLU(True),
        nn.Linear(392, 750),
        nn.ReLU(True),
        nn.Linear(750, 256*2*2)
        )

        self.conv_decoder_4 = nn.Sequential(
        nn.ConvTranspose2d(1, 256, kernel_size=4, stride=1, padding=1, output_padding=0),
        nn.BatchNorm2d(256),
        nn.ELU(),
        nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=2, output_padding=1),
        nn.BatchNorm2d(256),
        nn.Upsample(2),
        nn.ELU()
        )

        self.conv_decoder_1 = nn.Sequential(
        nn.ConvTranspose2d(256, 256, kernel_size=4, stride=1, padding=1, output_padding=0),
        nn.BatchNorm2d(256),
        nn.ELU(),
        nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=2, output_padding=1),
        nn.BatchNorm2d(128),
        nn.ELU()
        )

        self.shortcut_de1 = nn.Sequential(
        nn.ConvTranspose2d(256, 128, kernel_size=1, stride=2, padding=1, output_padding=0),
        nn.BatchNorm2d(128),
        nn.Upsample(5),
        nn.ELU()
        )

        self.conv_decoder_2 = nn.Sequential(
        nn.ConvTranspose2d(128, 128, kernel_size=4, stride=1, padding=1, output_padding=0),
        nn.BatchNorm2d(128),
        nn.ELU(),
        nn.ConvTranspose2d(128, 100, kernel_size=4, stride=2, padding=2, output_padding=1),
        nn.BatchNorm2d(100),
        nn.ELU()
        )

        self.shortcut_de2 = nn.Sequential(
        nn.ConvTranspose2d(128, 100, kernel_size=1, stride=2, padding=1, output_padding=0),
        nn.BatchNorm2d(100),
        nn.Upsample(11),
        nn.ELU()
        )

        self.conv_decoder_3 = nn.Sequential(
        nn.ConvTranspose2d(100, 100, kernel_size=4, stride=1, padding=2, output_padding=0),
        nn.BatchNorm2d(100),
        nn.ELU(),
        nn.ConvTranspose2d(100, 100, kernel_size=4, stride=2, padding=2, output_padding=1),
        nn.BatchNorm2d(100),
        nn.ELU(),
        nn.ConvTranspose2d(100, 100, kernel_size=4, stride=1, padding=1, output_padding=0),
        nn.BatchNorm2d(100),
        nn.ELU(),
        nn.ConvTranspose2d(100, 100, kernel_size=4, stride=1, padding=1, output_padding=0),
        nn.BatchNorm2d(100),
        nn.ELU()
        )

        self.shortcut_de3 = nn.Sequential(
        nn.ConvTranspose2d(100, 100, kernel_size=1, stride=2, padding=1, output_padding=0),
        nn.BatchNorm2d(100),
        nn.Upsample(21),
        nn.ELU()
        )

        self.q_z_dist = normal_dist
        self.loss_dist = bernoulli_loss

        #self.resnet_encoder = ResNet18(self.z_dim)
        #self.resnet_decoder = ResNet18inv(self.z_dim)

    def encode(self, x):
        # First go through the convolutional layers
        output = self.conv1(x)
        if self.arch == 'resnet':
            output = F.elu(self.conv_encoder_1(output) + self.shortcut_en1(output))
            output = F.elu(self.conv_encoder_2(output) + self.shortcut_en2(output))
            output = F.elu(self.conv_encoder_3(output) + self.shortcut_en3(output))
            output = output.view(-1, 256*2*2)
            # Go through the linear layers
            output = self.linear_encoder(output)
            output_len = len(output[0]) // 2
            return output[:,:output_len], output[:,output_len:]

        if self.arch == 'convlin':
            output = self.conv_encoder_1(output)
            output = self.conv_encoder_2(output)
            output = self.conv_encoder_3(output)
            output = output.view(-1, 256*2*2)
            # Go through the linear layers
            output = self.linear_encoder(output)
            output_len = len(output[0]) // 2
            return output[:,:output_len], output[:,output_len:]

        if self.arch == 'fcn':
            output = self.conv_encoder_1(output)
            output = self.conv_encoder_2(output)
            output = self.conv_encoder_3(output)
            output = self.conv_encoder_4(output)
            z_sqrt = int(np.sqrt(self.z_dim))
            return output[:, 0, :z_sqrt, :z_sqrt], output[:, 1, :z_sqrt, :z_sqrt]


    def reparameterize(self, q_z):
        return q_z.rsample()

    def decode(self, z):
        if self.arch == 'resnet':
            z = self.linear_decoder(z).view(z.size(0), 256, 2, 2)
            output = F.elu(self.conv_decoder_1(z) + self.shortcut_de1(z))
            output = F.elu(self.conv_decoder_2(output) + self.shortcut_de2(output))
            output = F.elu(self.conv_decoder_3(output) + self.shortcut_de3(output))
            return self.deconv1(output)

        if self.arch == 'convlin':
            z = self.linear_decoder(z).view(z.size(0), 256, 2, 2)
            output = self.conv_decoder_1(z)
            output = self.conv_decoder_2(output)
            output = self.conv_decoder_3(output)
            return self.deconv1(output)

        if self.arch == 'fcn':
            z_sqrt = int(np.sqrt(self.z_dim))
            z = self.conv_decoder_4(z.view(z.size(0), 1, z_sqrt, z_sqrt))
            output = self.conv_decoder_1(z)
            output = self.conv_decoder_2(output)
            output = self.conv_decoder_3(output)
            return self.deconv1(output)
        #return self.resnet_decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        q_z = self.q_z_dist(mu, logvar)
        z = self.reparameterize(q_z)
        x_hat = self.decode(z)
        p_x = self.loss_dist(x_hat)

        return x_hat, q_z, p_x, z
