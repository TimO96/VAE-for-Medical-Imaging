# -*- coding: utf-8 -*-
"""
Created on Wed May  8 13:01:22 2019

@author: Joris
"""
import torch.utils.data
from torch import nn, optim



class VAE(nn.Module):
    def __init__(self, layers, activation_layer, p_z, q_z, loss_dist):
        super(VAE, self).__init__()
        if layers[0] is None:
            self.conv_encoder = None
        else:
            self.conv_encoder = nn.Sequential(*layers[0])
            
        if layers[1] is None:
            self.encoder = None
        else:
            self.encoder = nn.Sequential(*layers[1])
            
        if layers[2] is None:
            self.decoder = None
        else:
            self.decoder = nn.Sequential(*layers[2])
            
        if layers[3] is None:
            self.conv_decoder = None
        else:
            self.conv_decoder = nn.Sequential(*layers[3])
            self.en_conv_1 = nn.Sequential(layers[4])
            self.en_conv_2 = nn.Sequential(layers[5])

        if (len(activation_layer[0]) == 0):
            self.activation_layer_1 = nn.Sequential()
        else:
            self.activation_layer_1 = nn.Sequential(*activation_layer[0])
            
        if (len(activation_layer[1]) == 0):
            self.activation_layer_2 = nn.Sequential()
        else:
            self.activation_layer_2 = nn.Sequential(*activation_layer[1])
            
        if (len(activation_layer[2]) == 0):
            self.activation_layer_3 = nn.Sequential()
        else:
            self.activation_layer_3 = nn.Sequential(*activation_layer[2])
        
        self.beta = 1
        self.p_z = p_z
        self.q_z = q_z
        self.loss_dist = loss_dist
        self.x = 0
        self.y = 0
        self.z = 0
        
    def encode(self, x):
        if self.conv_encoder is None:
            x = x.view(-1,784)
            out = self.encoder(x)
        elif self.encoder is None:
            out = self.conv_encoder(x)
            loc =  self.activation_layer_1(self.en_conv_1(out))
            scale =  self.activation_layer_2(self.en_conv_2(out))
            self.x, self.y, self.z = loc.size(1), loc.size(2), loc.size(2)
            return loc.view(-1, self.x * self.y * self.z), scale.view(-1, self.x * self.y * self.z)
        else:
            out = self.conv_encoder(x)
            self.x, self.y, self.z = out.size(1), out.size(2), out.size(2)
            
            out = out.view(-1, self.x * self.y * self.z)
            out = self.encoder(out)

        length_out = len(out[0]) // 2
        return self.activation_layer_1(out[:,:length_out]), self.activation_layer_2(out[:,length_out:])

    def reparameterize(self, q_z_given_x):
        return q_z_given_x.rsample()

    def decode(self, z):
        if self.conv_decoder is None:
            x_hat = self.decoder(z)
        elif self.decoder is None:
            z = z.view(-1,self.x , self.y , self.z)
            x_hat = self.conv_decoder(z)
            x_hat = x_hat.view(-1,x_hat.size(2) * x_hat.size(3))
        else:
            x_hat = self.decoder(z)
            x_hat = x_hat.view(-1,self.x , self.y , self.z)
            x_hat = self.conv_decoder(x_hat)
            x_hat = x_hat.view(-1,x_hat.size(2) * x_hat.size(3))
        return self.activation_layer_3(x_hat)
    
    def up_beta(self):
        self.beta += 1
        
    def set_beta(self, beta):
        self.beta = beta
        
    def down_beta(self):
        if self.beta != 0:
            self.beta -= 1

    def forward(self, x):
        loc, scale = self.encode(x)      

        q_z_given_x = self.q_z(loc, scale) # for KL divergence
        
        z = self.reparameterize(q_z_given_x)
        x_hat = self.decode(z)

        p_x_given_z = self.loss_dist(x_hat) # loss function/ distribution
        
        loss = self.loss_function(x_hat, x, q_z_given_x, p_x_given_z, z)
        return x_hat, loss
    
    def loss_function(self, x_hat, x,q_z_given_x, p_x_given_z, z):
        x = x.view(-1,784) # hardcoded for MNIST
        BCE = -p_x_given_z.log_prob(x)
        KLD = q_z_given_x.log_prob(z) - self.p_z.log_prob(z)
        #KLD = kl_divergence(q_z_given_x.base_dist, self.p_z.base_dist) # vervangen en werkend krijgen 
        #KLD = torch.sum(KLD.sum(len(p_z.event_shape)-1))
        return (BCE + self.beta * KLD).mean()