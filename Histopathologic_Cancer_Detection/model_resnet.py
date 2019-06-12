# -*- coding: utf-8 -*-
"""
Created on Wed May  8 13:01:22 2019

@author: Joris
"""
import torch.utils.data
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable



class VAE_RESN(nn.Module):
    def __init__(self, layers, activation_layer, p_z, q_z, size, loss_type):
        super(VAE_RESN, self).__init__()
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
        self.loss_func = layers[3]
        self.z_dim = layers[4]
        self.x_dim = layers[5]
        self.conv1 = nn.Sequential(
            nn.Conv2d( 3, 12, kernel_size= 4, stride= 2, padding= 3)
        )

        self.conv_encoder_1 = nn.Sequential(
            nn.Conv2d(12, 12, kernel_size= 6, stride= 1, padding= 0),
            nn.Conv2d(12, 42, kernel_size= 4, stride= 2, padding= 3),
            nn.ReLU(True),
            nn.Conv2d(42, 60, kernel_size= 6, stride= 1, padding= 0),
            nn.Conv2d(60, 96, kernel_size= 6, stride= 1, padding= 0),
            nn.ReLU(True)
        )

        self.shortcut_en1 = nn.Sequential(
            nn.Conv2d(12, 96, kernel_size= 5, stride= 3, padding= 0),
            nn.ReLU(True)
        )

        self.conv_encoder_2 = nn.Sequential(
            nn.Conv2d(96, 120, kernel_size= 5, stride= 1, padding= 0),
            nn.Conv2d(120, 144, kernel_size= 5, stride= 1, padding= 1),
            nn.ReLU(True),
            nn.Conv2d(144, 180, kernel_size= 5, stride= 1, padding= 1),
            nn.Conv2d(180, 200, kernel_size= 5, stride= 1, padding= 2),
            nn.ReLU(True),
        )

        self.shortcut_en2 = nn.Sequential(
            nn.Conv2d(96, 200, kernel_size=4, stride=2, padding=2),
            nn.ReLU(True)
        )

        self.conv_encoder_3 = nn.Sequential(
            nn.Conv2d(200, 220, kernel_size= 5, stride= 1, padding= 1),
            nn.Conv2d(220, 256, kernel_size= 4, stride= 2, padding= 3),
            nn.ReLU(True)
        )

        self.shortcut_en3 = nn.Sequential(
            nn.Conv2d(200, 256, kernel_size=4, stride=2, padding=2),
            nn.ReLU(True)
        )

        self.linear_encoder = nn.Sequential(
            nn.Linear(self.x_dim, 8192),
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
            nn.Linear(192, self.z_dim*2)
        )

        self.linear_decoder = nn.Sequential(
            nn.Linear(self.z_dim, 128),
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
            nn.Linear(8192, self.x_dim)
        )

        self.conv_decoder_1 = nn.Sequential(
            nn.ConvTranspose2d(256, 220, kernel_size= 4, stride= 2, padding= 3, output_padding= 0),
            nn.ConvTranspose2d(220, 200, kernel_size= 5, stride= 1, padding= 1, output_padding= 0),
            nn.ReLU(True)
        )

        self.shortcut_de1 = nn.Sequential(
            nn.ConvTranspose2d(256, 200, kernel_size=4, stride=2, padding=2, output_padding=0),
            nn.ReLU(True)
        )

        self.conv_decoder_2 = nn.Sequential(
            nn.ConvTranspose2d(200, 180, kernel_size= 5, stride= 1, padding= 2, output_padding= 0),
            nn.ConvTranspose2d(180, 144, kernel_size= 5, stride= 1, padding= 1, output_padding= 0),
            nn.ReLU(True),
            nn.ConvTranspose2d(144, 120, kernel_size= 5, stride= 1, padding= 1, output_padding= 0),
            nn.ConvTranspose2d(120, 96, kernel_size= 5, stride= 1, padding= 0, output_padding= 0),
            nn.ReLU(True)
        )

        self.shortcut_de2 = nn.Sequential(
            nn.ConvTranspose2d(200, 96, kernel_size=4, stride=2, padding=2, output_padding=0),
            nn.ReLU(True)
        )
        if self.loss_func == 1:
            self.conv_decoder_3 = nn.Sequential(
                nn.ConvTranspose2d(96, 60, kernel_size= 6, stride= 1, padding= 0, output_padding= 0),
                nn.ConvTranspose2d(60, 42, kernel_size= 6, stride= 1, padding= 0, output_padding= 0),
                nn.ConvTranspose2d(42, 12, kernel_size= 4, stride= 2, padding= 3, output_padding= 1),
                nn.ReLU(True),
                nn.ConvTranspose2d(12, 12, kernel_size= 6, stride= 1, padding= 0, output_padding= 0),
                nn.ConvTranspose2d(12,  100, kernel_size= 4, stride= 2, padding= 3, output_padding= 0)
            )
            
            self.shortcut_de3 = nn.Sequential(
                nn.ConvTranspose2d(96, 12, kernel_size= 5, stride= 3, padding= 0, output_padding= 2),
                nn.ConvTranspose2d(12,  100, kernel_size= 4, stride= 2, padding= 3, output_padding= 0),
                nn.ReLU(True)
                
            )
        else:
            self.conv_decoder_3 = nn.Sequential(
                nn.ConvTranspose2d(96, 60, kernel_size= 6, stride= 1, padding= 0, output_padding= 0),
                nn.ConvTranspose2d(60, 42, kernel_size= 6, stride= 1, padding= 0, output_padding= 0),
                nn.ConvTranspose2d(42, 12, kernel_size= 4, stride= 2, padding= 3, output_padding= 1),
                nn.ReLU(True),
                nn.ConvTranspose2d(12, 12, kernel_size= 6, stride= 1, padding= 0, output_padding= 0),
                nn.ConvTranspose2d(12,  3*256, kernel_size= 4, stride= 2, padding= 3, output_padding= 0)
            )
            
            self.shortcut_de3 = nn.Sequential(
                nn.ConvTranspose2d(96, 12, kernel_size= 5, stride= 3, padding= 0, output_padding= 2),
                nn.ConvTranspose2d(12,  3*256, kernel_size= 4, stride= 2, padding= 3, output_padding= 0)
                
            )
            
        self.beta = 1
        self.p_z = p_z
        self.q_z = q_z
        self.batch = 128
        self.x = 0
        self.y = 0
        self.z = 0
        self.size = size * size
        self.loss_type = loss_type
        self.loss = nn.CrossEntropyLoss()
        
    def encode(self, x):
        output = self.conv1(x)
        output = F.elu(self.conv_encoder_1(output) + self.shortcut_en1(output))
        output = F.elu(self.conv_encoder_2(output) + self.shortcut_en2(output))
        output = F.elu(self.conv_encoder_3(output) + self.shortcut_en3(output))
        self.x, self.y, self.z = output.size(1), output.size(2), output.size(2)
        output = output.view(-1, self.x * self.y * self.z)
        output = self.linear_encoder(output)

        length_out = len(output[0]) // 2
        return self.activation_layer_1(output[:,:length_out]), self.activation_layer_2(output[:,length_out:])

    def reparameterize(self, q_z_given_x):
        return q_z_given_x.rsample()

    def decode(self, z):
        x_hat = self.linear_decoder(z)
        x_hat = x_hat.view(-1,self.x , self.y , self.z)
        x_hat = F.elu(self.conv_decoder_1(x_hat) + self.shortcut_de1(x_hat))
        x_hat = F.elu(self.conv_decoder_2(x_hat) + self.shortcut_de2(x_hat))
        x_hat = F.elu(self.conv_decoder_3(x_hat) + self.shortcut_de3(x_hat))
#        x_hat = x_hat.view(-1,x_hat.size(2) * x_hat.size(3))
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
        
        loss = self.loss_function(x_hat, x, q_z_given_x, z)
        return x_hat, loss
    
    def loss_function(self, x_hat, x,q_z_given_x, z):
#        x = x.view(-1, self.size) # hardcoded for MNIST
#        BCE = -p_x_given_z.log_prob(x)
        if self.loss_type:
            x_hat= x_hat.view(self.batch,100,112,112)
            recon_loss = -self.log_mix_dep_Logistic_256(x, x_hat, average=True, n_comps=10)
        else:
            x_hat = x_hat.view(self.batch, 3, 256, 112, 112)
            x_hat = Variable(x_hat)
            x_hat = x_hat.permute(0, 1, 3, 4, 2)
            x_hat = x_hat.contiguous()
            x_hat = torch.round(256 * x_hat.view(-1, 256))
            target = Variable(x.data.view(-1) * 255).long()
            recon_loss = self.loss(x_hat, target)
            
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
    
    def test_encode_decode(self, image):
        loc, scale = self.encode(image) 

        q_z_given_x = self.q_z(loc, scale) # for KL divergence
        
        z = self.reparameterize(q_z_given_x)
        x_hat = self.decode(z)
        print(x_hat.shape, z.shape)
        
        loss = self.loss_function(x_hat, image, q_z_given_x, z)
        print(loss)
        print()