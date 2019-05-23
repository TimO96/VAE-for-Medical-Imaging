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

def laplace_loss(x_hat, scale=0.025):
    return Laplace(loc=x_hat, scale=scale)

class VAE(nn.Module):
    def __init__(self, x_dim, z_dim):
        super(VAE, self).__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim

        self.conv1 = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=4, stride=1, padding=2),
        nn.ELU()
        )

        self.deconv1 = nn.Sequential(
        nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1, output_padding=0),
        nn.Upsample(28),
        nn.Sigmoid()
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
        nn.Conv2d(32, 64, kernel_size=1, stride=6, padding=0),
        nn.BatchNorm2d(64),
        nn.ELU()
        )

        self.conv_encoder_2 = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=2),
        nn.BatchNorm2d(128),
        nn.ELU(),
        nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=2),
        nn.BatchNorm2d(128),
        #nn.ELU()
        )

        self.shortcut_en2 = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=1, stride=6, padding=0),
        nn.BatchNorm2d(128),
        #nn.ELU()
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
        nn.Conv2d(128, 256, kernel_size=1, stride=2, padding=0),
        nn.BatchNorm2d(256),
        nn.ELU()
        )

        self.linear_encoder = nn.Sequential(
        nn.Linear(256*2*2, 700),
        nn.ReLU(True),
        nn.Linear(700, 392),
        nn.ReLU(True),
        nn.Linear(392, 100),
        nn.ReLU(True),
        nn.Linear(100, 49),
        nn.ReLU(True),
        nn.Linear(49, self.z_dim*2),
        nn.Softplus()
        )

        self.linear_decoder = nn.Sequential(
        nn.Linear(self.z_dim, 49),
        nn.ReLU(True),
        nn.Linear(49, 100),
        nn.ReLU(True),
        nn.Linear(100, 392),
        nn.ReLU(True),
        nn.Linear(392, 700),
        nn.ReLU(True),
        nn.Linear(700, 256*2*2),
        nn.Sigmoid()
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
        nn.ConvTranspose2d(256, 128, kernel_size=1, stride=4, padding=0, output_padding=0),
        nn.BatchNorm2d(128),
        nn.ELU()
        )

        self.conv_decoder_2 = nn.Sequential(
        nn.ConvTranspose2d(128, 128, kernel_size=4, stride=1, padding=1, output_padding=0),
        nn.BatchNorm2d(128),
        nn.ELU(),
        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=2, output_padding=1),
        nn.BatchNorm2d(64),
        nn.ELU()
        )

        self.shortcut_de2 = nn.Sequential(
        nn.ConvTranspose2d(128, 64, kernel_size=1, stride=3, padding=1, output_padding=0),
        nn.BatchNorm2d(64),
        nn.ELU()
        )

        self.conv_decoder_3 = nn.Sequential(
        nn.ConvTranspose2d(64, 64, kernel_size=4, stride=1, padding=2, output_padding=0),
        nn.BatchNorm2d(64),
        nn.ELU(),
        nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=2, output_padding=1),
        nn.BatchNorm2d(32),
        nn.ELU(),
        nn.ConvTranspose2d(32, 32, kernel_size=4, stride=1, padding=1, output_padding=0),
        nn.BatchNorm2d(32),
        nn.ELU(),
        nn.ConvTranspose2d(32, 32, kernel_size=4, stride=1, padding=1, output_padding=0),
        nn.BatchNorm2d(32),
        #nn.ELU()
        )

        self.shortcut_de3 = nn.Sequential(
        nn.ConvTranspose2d(64, 32, kernel_size=1, stride=2, padding=0, output_padding=0),
        nn.BatchNorm2d(32),
        #nn.ELU()
        )

        self.q_z_dist = normal_dist
        self.loss_dist = laplace_loss

        #self.resnet_encoder = ResNet18(self.z_dim)
        #self.resnet_decoder = ResNet18inv(self.z_dim)

    def encode(self, x):
        # First go through the convolutional layers

        output = self.conv1(x)
        output = F.elu(self.conv_encoder_1(output) + self.shortcut_en1(output))
        output = F.elu(self.conv_encoder_2(output) + self.shortcut_en2(output))
        output = F.elu(self.conv_encoder_3(output) + self.shortcut_en3(output))
        #output = self.resnet_encoder(x)
        # Flatten the output (TODO: make 4*4*50 a variable that comes from the convolutional layers)
        output = output.view(-1, 256*2*2)
        # Go through the linear layers
        output = self.linear_encoder(output)
        output_len = len(output[0]) // 2
        return output[:,:output_len], output[:,output_len:]


    def reparameterize(self, q_z):
        return q_z.rsample()

    def decode(self, z):
        z = self.linear_decoder(z).view(z.size(0), 256, 2, 2)
        output = F.elu(self.conv_decoder_1(z) + self.shortcut_de1(z))
        output = F.elu(self.conv_decoder_2(output) + self.shortcut_de2(output))
        output = F.elu(self.conv_decoder_3(output) + self.shortcut_de3(output))
        return self.deconv1(output)
        #return self.resnet_decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        q_z = self.q_z_dist(mu, logvar)
        z = self.reparameterize(q_z)
        x_hat = self.decode(z)
        p_x = self.loss_dist(x_hat.view(x_hat.size(0), self.x_dim))

        return x_hat, q_z, p_x, z
