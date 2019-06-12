# -*- coding: utf-8 -*-
"""
Created on Thu May  9 17:08:31 2019

@author: Joris
"""

from torch.distributions import Normal, Laplace, Independent, Bernoulli, Gamma, Uniform, Beta

class distributions():
    def normal(self, loc, scale):
         return Independent(Normal(loc=loc, scale=scale), 1)
    
    def laplace(self, loc, scale):
        return Independent(Laplace(loc=loc, scale=scale), 1)
    
    def gamma(self, loc, scale):
        return Independent(Gamma(loc, scale), 1)
    
    def beta(self, loc, scale):
        return Independent(Beta(loc, scale), 1)
    
    def bernoulli_loss(self, x_hat):
        return Independent(Bernoulli(x_hat), 1)
    
    def laplace_loss(self, x_hat):
        return Independent(Laplace(loc=x_hat, scale=1e-2), 1)
    
    def normal_loss(self, x_hat):
        return Independent(Normal(loc=x_hat, scale=1), 1)   