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
    def __init__(self, x_dim, z_dim):
        super(VAE, self).__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim

        '''
        self.conv_encoder = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=4, stride=1, padding=2),
        nn.ELU(),
        nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=2),
        nn.ELU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=2),
        nn.ELU(),
        nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=2),
        nn.ELU(),
        nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=2),
        nn.ELU(),
        nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=0),
        )

        self.linear_encoder = nn.Sequential(
        nn.Linear(128*16*16, 1000),
        nn.ReLU(True),
        nn.Linear(1000, 392),
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
        nn.Linear(392, 1000),
        nn.ReLU(True),
        nn.Linear(1000, 128*16*16),
        nn.Sigmoid()
        )

        self.conv_decoder = nn.Sequential(
        nn.ConvTranspose2d(128, 128, kernel_size=4, stride=1, padding=1, output_padding=0),
        nn.ELU(),
        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=2, output_padding=1),
        nn.ELU(),
        nn.ConvTranspose2d(64, 64, kernel_size=4, stride=1, padding=2, output_padding=0),
        nn.ELU(),
        nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=2, output_padding=1),
        nn.ELU(),
        nn.ConvTranspose2d(32, 32, kernel_size=4, stride=1, padding=1, output_padding=0),
        nn.ELU(),
        nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1, output_padding=0),
        nn.Sigmoid()
        )

        '''

        class BasicBlock(nn.Module):
            expansion = 1

            def __init__(self, in_planes, planes, stride=1):
                super(BasicBlock, self).__init__()
                self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(planes)
                self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(planes)

                self.shortcut = nn.Sequential()
                if stride != 1 or in_planes != self.expansion*planes:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm2d(self.expansion*planes)
                    )

            def forward(self, x):
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out += self.shortcut(x)
                out = F.relu(out)
                return out

        class BasicBlockinv(nn.Module):
            expansion = 1

            def __init__(self, in_planes, planes, stride=1):
                super(BasicBlockinv, self).__init__()
                self.conv1 = nn.ConvTranspose2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, output_padding=stride-1, bias=False)
                self.bn1 = nn.BatchNorm2d(planes)
                self.conv2 = nn.ConvTranspose2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(planes)

                self.shortcut = nn.Sequential()
                if stride != 1 or in_planes != self.expansion*planes:
                    self.shortcut = nn.Sequential(
                        nn.ConvTranspose2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, output_padding=stride-1, bias=False),
                        nn.BatchNorm2d(self.expansion*planes)
                    )

            def forward(self, x):
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out += self.shortcut(x)
                out = F.relu(out)
                return out


        class ResNet(nn.Module):
            def __init__(self, block, num_blocks, num_classes=20):
                super(ResNet, self).__init__()
                self.in_planes = 64

                self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(64)
                self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
                self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
                self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
                self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
                self.linear = nn.Sequential(nn.Linear(512*block.expansion, num_classes), nn.Softplus())

            def _make_layer(self, block, planes, num_blocks, stride):
                strides = [stride] + [1]*(num_blocks-1)
                layers = []
                for stride in strides:
                    layers.append(block(self.in_planes, planes, stride))
                    self.in_planes = planes * block.expansion
                return nn.Sequential(*layers)

            def forward(self, x):
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.layer1(out)
                out = self.layer2(out)
                out = self.layer3(out)
                out = self.layer4(out)
                out = F.avg_pool2d(out, 4)
                out = out.view(out.size(0), -1)
                out = self.linear(out)
                return out

        class ResNetinv(nn.Module):
            def __init__(self, block, num_blocks, num_classes=10):
                super(ResNetinv, self).__init__()
                self.in_planes = 512

                self.linear = nn.Linear(num_classes, 512*16*block.expansion)
                self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
                self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
                self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
                self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
                self.conv1 = nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(64)

            def _make_layer(self, block, planes, num_blocks, stride):
                strides = [stride] + [1]*(num_blocks-1)
                layers = []
                for stride in strides:
                    layers.append(block(self.in_planes, planes, stride))
                    self.in_planes = planes * block.expansion
                return nn.Sequential(*layers)

            def forward(self, x):
                out = self.linear(x)
                out = out.view(out.size(0), 512, 4, 4)
                out = self.layer4(out)
                out = self.layer3(out)
                out = self.layer2(out)
                out = self.layer1(out)
                out = F.relu(self.conv1(out))

                return out

        def ResNet18(z_dim):
            return ResNet(BasicBlock, [2,2,2,2], z_dim*2)

        def ResNet18inv(z_dim):
            return ResNetinv(BasicBlockinv, [2,2,2,2], z_dim)


        self.q_z_dist = beta_dist
        self.loss_dist = laplace_loss
        self.resnet_encoder = ResNet18(self.z_dim)
        self.resnet_decoder = ResNet18inv(self.z_dim)

    def encode(self, x):
        # First go through the convolutional layers
        output = self.resnet_encoder(x)
        # Flatten the output (TODO: make 4*4*50 a variable that comes from the convolutional layers)
        #output = output.view(-1, 128*16*16)
        # Go through the linear layers
        #output = self.linear_encoder(output)
        output_len = len(output[0]) // 2
        return output[:,:output_len], output[:,output_len:]


    def reparameterize(self, q_z):
        return q_z.rsample()

    def decode(self, z):
        return self.resnet_decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        q_z = self.q_z_dist(mu, logvar)
        z = self.reparameterize(q_z)
        x_hat = self.decode(z)
        p_x = self.loss_dist(x_hat)

        return x_hat, q_z, p_x, z
