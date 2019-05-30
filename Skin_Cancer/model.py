# -*- coding: utf-8 -*-
"""
Created on Wed May  8 13:01:22 2019

@author: Joris
"""
import torch.utils.data
import torch.nn.functional as F
from torch import nn, optim



class VAE(nn.Module):
    def __init__(self, layers, activation_layer, p_z, q_z, loss_dist, size):
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
        self.batch = 0
        self.x = 0
        self.y = 0
        self.z = 0
        self.size = size * size
        
    def encode(self, x):
        if self.conv_encoder is None:
            x = x.view(-1, self.size)
            out = self.encoder(x)
        elif self.encoder is None:
            out = self.conv_encoder(x)
            loc =  self.activation_layer_1(self.en_conv_1(out))
            scale =  self.activation_layer_2(self.en_conv_2(out))
            self.batch = loc.size(0)
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
    
    def get_z(self, x):
        loc, scale = self.encode(x)      

        q_z_given_x = self.q_z(loc, scale) # for KL divergence
        
        z = self.reparameterize(q_z_given_x)
        return z

    def forward(self, x):
        loc, scale = self.encode(x)      

        q_z_given_x = self.q_z(loc, scale) # for KL divergence
        
        z = self.reparameterize(q_z_given_x)
        x_hat = self.decode(z)

        p_x_given_z = self.loss_dist(x_hat) # loss function/ distribution
        
        loss = self.loss_function(x_hat, x, q_z_given_x, p_x_given_z, z)
        return x_hat, loss
    
    def loss_function(self, x_hat, x,q_z_given_x, p_x_given_z, z):
#        x = x.view(-1, self.size) # hardcoded for MNIST
#        BCE = -p_x_given_z.log_prob(x)
        x_hat= x_hat.view(self.batch,100,64,64)
        recon_loss = -self.log_mix_dep_Logistic_256(x, x_hat, average=True, n_comps=10)
        
        KLD = q_z_given_x.log_prob(z) - self.p_z.log_prob(z)
        #KLD = kl_divergence(q_z_given_x.base_dist, self.p_z.base_dist) # vervangen en werkend krijgen 
        #KLD = torch.sum(KLD.sum(len(p_z.event_shape)-1))
        return (recon_loss + self.beta * KLD).mean()
    
    def log_sum_exp(self, x):
        """ numerically stable log_sum_exp implementation that prevents overflow """
        # TF ordering
        axis = len(x.size()) - 1
        m, _ = torch.max(x, dim=axis)
        m2, _ = torch.max(x, dim=axis, keepdim=True)
        return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))
    
    def log_mix_dep_Logistic_256(self, x, params, average=False, n_comps=10):
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
        log_logist_256 = self.log_sum_exp(log_logist_256)
    
        # flatten to [B, H * W]
        log_logist_256 = log_logist_256.view(log_logist_256.size(0), -1)
    
        # if reduce:
        if average:
            return torch.mean(log_logist_256, 1)
        else:
            return torch.sum(log_logist_256)
            # else:
        #     return log_logist_256