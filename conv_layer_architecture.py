# -*- coding: utf-8 -*-
"""
Created on Wed May  8 13:51:31 2019

@author: Joris
"""
import numpy as np
import torch
from torch import nn, optim
import itertools

class conv_layer_architecture():
    def __init__(self, architecture, min_k, max_k, max_s, max_p):
        self.max_k = max_k
        self.min_k = min_k
        self.max_s = max_s
        self.max_p = max_p
        self.architecture = self.get_parameters(architecture)
        self.reverse_architecture = self.flip_architecture()
        self.adjust_decoder()
        
    def report(self):
        start_size = self.architecture[0,2]
        test_layer = torch.zeros((1, 1, start_size,start_size))
        print("\nEncoder architecture test")
        print(test_layer.shape)
        for i, (row) in enumerate(self.architecture):
             layer = nn.Sequential(nn.Conv2d(row[0], row[1], kernel_size=row[4], stride=row[5], padding=row[6]))
             test_layer = layer(test_layer)
             print(test_layer.shape)
        
        print("\nDecoder architecture test")
        for i, (row) in enumerate(self.reverse_architecture):
             layer = nn.Sequential(nn.ConvTranspose2d(row[0], row[1], kernel_size=row[4], stride=row[5], padding=row[6], output_padding=row[7]))
             test_layer = layer(test_layer)
             print(test_layer.shape)
        
        print("\nEncoder architecture parameters")
        for layer in self.architecture:     
            print("Channel_in=%3d Channel_out=%3d Image_Size_in=%3d Image_Size_out=%3d Kernel_Size=%2d Stride=%2d Padding=%2d" 
                  % (layer[0], layer[1], layer[2], layer[3], layer[4], layer[5], layer[6]))
        
        print("\nDecoder architecture parameters")
        for layer in self.reverse_architecture:     
            print("Channel_in=%3d Channel_out=%3d Image_Size_in=%3d Image_Size_out=%3d Kernel_Size=%2d Stride=%2d Padding=%2d output_padding=%2d" 
                  % (layer[0], layer[1], layer[2], layer[3], layer[4], layer[5], layer[6], layer[7]))
            
    def flip_architecture(self):
        decoder_architecture = self.architecture.copy()
        decoder_architecture = np.flip(decoder_architecture, 0)
        decoder_architecture[:, 2:4] = np.flip(decoder_architecture[:, 2:4], 1)
        decoder_architecture[:, 0:2] = np.flip(decoder_architecture[:, 0:2], 1)
        return np.append(decoder_architecture, np.zeros((len(self.architecture),1)), 1)
    
    def adjust_decoder(self):
        start_size = self.architecture[0,2]
        test_layer = torch.zeros((1, 1, start_size,start_size))
        sizes_encoder = np.zeros((len(self.architecture), 2))
        for i, (row) in enumerate(self.architecture):
             layer = nn.Sequential(nn.Conv2d(row[0], row[1], kernel_size=row[4], stride=row[5], padding=row[6]))
             test_layer = layer(test_layer)
             sizes_encoder[i] = test_layer.shape[2:4]
             
        sizes_encoder = np.flip(sizes_encoder[:-1])
        sizes_decoder = self.get_sizes_decoder(test_layer.clone())
        for i, (row) in enumerate(sizes_encoder):
            while sizes_decoder[i, 0] < row[0]:
                self.reverse_architecture[i, 7] += 1
                sizes_decoder = self.get_sizes_decoder(test_layer.clone())
        
    def get_sizes_decoder(self, test_layer):
        sizes_decoder = np.zeros((len(self.architecture), 2))
        for i, (row) in enumerate(self.reverse_architecture):
             row[7] = int(row[7])
             layer = nn.Sequential(nn.ConvTranspose2d(row[0], row[1], kernel_size=row[4], stride=row[5], padding=row[6], output_padding=row[7]))
             test_layer = layer(test_layer)
             sizes_decoder[i] = test_layer.shape[2:4]
             
        return sizes_decoder
        
        
    def get_parameters(self, architecture):
        for i, (row) in enumerate(architecture):
            if row[4] is None or row[5] is None or row[6] is None:
                row[4], row[5], row[6] = self.calc_parameters(row)
                row[3] = self.calc_output(row)
                if i < len(architecture) - 1:
                    architecture[i + 1, 2] = row[3]
            elif row[3] is None:
                row[3] = self.calc_output(row)
                
        return architecture
        
    def calc_parameters(self, parameters):
        w = parameters[2]    
        o = parameters[3]
        k = parameters[4] 
        s = parameters[5] 
        p = parameters[6] 
        
        parameter_names = {}
        parameter_options = []
        if k is None:
            k_options = np.arange(self.min_k, self.max_k + 1)
            parameter_options.append(list(k_options))
            parameter_names.update({"k":len(parameter_options) - 1})
            if s is None:
                s_options = np.arange(1, self.max_s + 1)
                parameter_options.append(list(s_options))
                parameter_names.update({"s":len(parameter_options) - 1})
            if p is None:
                p_options = np.arange(0, self.max_p + 1)
                parameter_options.append(list(p_options))
                parameter_names.update({"p":len(parameter_options) - 1})
        else:
            if s is None:
                s_options = np.arange(1, self.max_s + 1)
                parameter_options.append(list(s_options))
                parameter_names.update({"s":len(parameter_options) - 1})
            if p is None:
                p_options = np.arange(0,self.max_p + 1)
                parameter_options.append(list(p_options))
                parameter_names.update({"p":len(parameter_options) - 1})
        
        lenght = 1
        for i in parameter_options:
            lenght *= len(i)
            
        options = np.zeros((lenght, 4))
        combinations = np.array(list(itertools.product(*parameter_options)))
        
        for i, (combination) in enumerate(combinations):
            if "k" in parameter_names:
                k = combination[parameter_names["k"]]
            if "s" in parameter_names:
                s = combination[parameter_names["s"]]
            if "p" in parameter_names:
                p = combination[parameter_names["p"]]
                
            options[i] = np.array([k, s, p, int(((w - k + 2 * p) / s) + 1)])
        
        pick = np.argmin(abs(options[:,3] - o))
        return int(options[pick,0]), int(options[pick,1]), int(options[pick,2])
    
    def calc_output(self, parameters):
        w = parameters[2] 
        k = parameters[4] 
        s = parameters[5] 
        p = parameters[6] 
        return int(((w - k + 2 * p) / s) + 1)
    
    def get_conv_text(self):
        print("\nEncoder architecture")
        for layer in self.architecture[:-1]:
            print("nn.Conv2d(%2d, %2d, kernel_size=%2d, stride=%2d, padding=%2d)," % (layer[0], layer[1], layer[4], layer[5], layer[6]))
        
        print("\nEncoder parameter architecture")
        for layer in self.architecture[-1:]:
            print("nn.Conv2d(%2d, %2d, kernel_size=%2d, stride=%2d, padding=%2d)," % (layer[0], layer[1], layer[4], layer[5], layer[6]))

        print("\nDecoder architecture")
        for layer in self.reverse_architecture:
            print("nn.ConvTranspose2d(%2d, %2d, kernel_size=%2d, stride=%2d, padding=%2d, output_padding=%2d)," % (layer[0], layer[1], layer[4], layer[5], layer[6], layer[7]))
            
            
if __name__ == "__main__":   
    # ------------------------------------------------------------------------------------------     
    # o = output
    # w = input
    # k = filter
    # p = padding
    # s = stride    
    # [1, 1, 28, 28, None, None, None]
    # Set up as [Channel_in, Channel_out, Image_Size_in, Image_Size_out, Kernel_Size, Stride, Padding]
    # Note that the decoder has one more parameter, output_padding
    # ------------------------------------------------------------------------------------------
    # If you want the leave the output size open (None) please fill in all parameters
    # If you you fill in the output size, you can leave any combination of kernel size, stride or padding empty
    # It will try to fill it in, so that it matches the desired output size as close as possible
    # ------------------------------------------------------------------------------------------
    conv_architecture = np.array([[1, 10, 28, 28, None, None, None],
                                  [10, 20, 24, 24, None, None, None],
                                  [20, 30, 20, 20, None, None, None],
                                  [30, 40, 16, 16, None, None, None],
                                  [40, 50, 12, 12,  None, None, None],
                                  [50, 60, 12, 12,  None, None, None],
                                  [60, 64, 4, 4, None, None, None],
                                  [64, 64, 4, 4, None, 2, None]])
    # ------------------------------------------------------------------------------------------
    # set the intervall between which the program is allowed to search
    # stride is set between 1 and max_stride
    # padding is set between 0 and max_padding
    # ------------------------------------------------------------------------------------------
    min_kernel_size = 4
    max_kernel_size = 5 
    max_stride = 3      
    max_padding = 5     
    architecture = conv_layer_architecture(conv_architecture, min_kernel_size, max_kernel_size, max_stride, max_padding)
    architecture.report()
    architecture.get_conv_text()            
  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
