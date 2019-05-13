from __future__ import print_function
import argparse
import torch
import torch.utils.data
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

def laplace_loss(x_hat, scale=0.01):
    return Laplace(loc=x_hat, scale=scale)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.x_dim = 784
        self.z_dim = 20

        self.conv_encoder = nn.Sequential(
        nn.Conv2d(1, 20, 5, 1),
        nn.ReLU(True),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(20, 50, 5, 1),
        nn.ReLU(True),
        nn.MaxPool2d(2, 2)
        )

        self.linear_encoder = nn.Sequential(
        nn.Linear(50*4*4, 392),
        nn.ReLU(True),
        nn.Linear(392, 196),
        nn.ReLU(True),
        nn.Linear(196, 49),
        nn.ReLU(True),
        nn.Linear(49, self.z_dim*2),
        nn.Softplus()
        )

        self.linear_decoder = nn.Sequential(
        nn.Linear(self.z_dim, 49),
        nn.ReLU(True),
        nn.Linear(49, 196),
        nn.ReLU(True),
        nn.Linear(196, 392),
        nn.ReLU(True),
        nn.Linear(392, 50*4*4),
        nn.Sigmoid()
        )

        self.conv_decoder = nn.Sequential(
        nn.ConvTranspose2d(50, 20, 5, 1),
        nn.ReLU(True),
        nn.Upsample(scale_factor=3),
        nn.ConvTranspose2d(20, 1, 5, 1),
        nn.ReLU(True)
        )

        tensor = torch.ones(1)
        self.p_x_dist = Beta(tensor.new_full((1, 20), 0.5), tensor.new_full((1, 20), 0.5))
        self.p_x_dist = Independent(self.p_x_dist, 1)

        self.q_z_dist = beta_dist
        self.loss_dist = bernoulli_loss

    def encode(self, x):
        # First go through the convolutional layers
        output = self.conv_encoder(x)
        # Flatten the output (TODO: make 4*4*50 a variable that comes from the convolutional layers)
        output = output.view(-1, 4*4*50)
        # Go through the linear layers
        output = self.linear_encoder(output)
        output_len = len(output[0]) // 2
        return output[:,:output_len], output[:,output_len:]

    def reparameterize(self, q_z):
        return q_z.rsample()

    def decode(self, z):
        z = self.linear_decoder(z).view(z.size(0), 50, 4, 4)
        return self.conv_decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)

        q_z = self.q_z_dist(mu, logvar)
        q_z = Independent(q_z, 1)

        z = self.reparameterize(q_z)
        x_hat = self.decode(z).view(-1, self.x_dim)

        p_x = self.loss_dist(x_hat)

        return x_hat, p_x, q_z, self.p_x_dist, z
