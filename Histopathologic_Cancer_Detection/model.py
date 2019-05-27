import torch.nn as nn
from loader import *

class VAE(nn.Module):
    def __init__(self, conv_encoder, linear_encoder, linear_decoder,
    conv_decoder, loss_dist, p_x_dist, q_z_dist, max_beta, one_third_epochs):
        super(VAE, self).__init__()
        self.conv_encoder = nn.Sequential(*conv_encoder)
        self.linear_encoder = nn.Sequential(*linear_encoder)
        self.linear_decoder = nn.Sequential(*linear_decoder)
        self.conv_decoder = nn.Sequential(*conv_decoder)
        self.p_x_dist = p_x_dist
        self.q_z_dist = q_z_dist
        self.loss_dist = loss_dist
        self.max_beta = max_beta
        self.one_third_epochs = one_third_epochs

    def encode(self, x):
        # First go through the convolutional layers
        #output = self.conv_encoder(x)
        # Flatten the output (TODO: make 4*4*50 a variable that comes from the convolutional layers)
        #output = output.view(-1, 4*4*64)
        # Go through the linear layers
        #output = self.linear_encoder(output)
        x = x.view(-1, 96*96*3)
        output = self.linear_encoder(x)
        output_len = len(output[0]) // 2
        return output[:,:output_len], output[:,output_len:]

    def reparameterize(self, q_z):
        return q_z.rsample()

    def decode(self, z):
        z = self.linear_decoder(z)#.view(z.size(0), 64, 4, 4)
        return self.conv_decoder(z)

    def forward(self, x, epoch):
        mu, logvar = self.encode(x)

        q_z = self.q_z_dist(mu, logvar)
        q_z = Independent(q_z, 1)

        z = self.reparameterize(q_z)
        x_hat = self.decode(z)#.view(-1, x_dim)

        p_x = self.loss_dist(x_hat)
        loss = self.loss_function(x_hat, x, p_x, q_z, z, epoch)

        return x_hat, loss, z

    def loss_function(self, x_hat, x, p_x, q_z, z, epoch):
        x = x.view(-1, 96*96*3)
        BCE = torch.sum(-p_x.log_prob(x))
        KLD = kl_divergence(q_z.base_dist, self.p_x_dist.base_dist)
        KLD = torch.sum(KLD.sum(len(self.p_x_dist.event_shape)-1))

        if epoch < self.one_third_epochs:
            beta = epoch * self.max_beta / self.one_third_epochs

        else:
            beta = self.max_beta

        return BCE + beta*KLD
