# Copyright (c) 2010-2025, InterDigital
# All rights reserved. 
# See LICENSE under the root folder.

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
import numpy as np

class Weights_Model(nn.Module):
    def __init__(self,config):
        super(Weights_Model, self).__init__()

        self.config = config

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.1))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers


        modules = []
        hidden_dims = [32, 64, 128, 256, 512,1024]
        in_channels = 2
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 2, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.lin1 = nn.Linear(2048,1024)
        self.lin2 = nn.Linear(1024,256)
        self.lin3 = nn.Linear(256,40)
        
        self.model = nn.Sequential(*modules)

        '''self.bottleneck = nn.Sequential(
            nn.Linear(2048,2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048,2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048,2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048,2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048,2048),
            nn.LeakyReLU(0.2),
        )'''

        self.bottleneck = nn.Sequential(
            nn.Linear(4096,4096),
            nn.LeakyReLU(0.2),
        )
        
        self.decoder_hidden_dims = [512,256,128,64,32,16]

        def conv_block(in_feat,out_feat,k_size,stride,padding=0):
            layers = [nn.ConvTranspose2d(in_feat,out_feat,k_size,stride=stride,padding=padding)]
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # For Resolution 20 -> Kernel Size = 4

        kernel_size = 4
        stride = 1
        padding = 0
        self.decoder = nn.Sequential(                                # For Kernel Size = 4
            *conv_block(512,256,kernel_size,stride,padding),                    # Output : 4 x 4
            *conv_block(256,128,kernel_size,stride,padding),                    # Output : 8 x 8
            *conv_block(128,64,kernel_size,stride,padding),                     # Output : 16 x 16
            *conv_block(64,64,kernel_size,stride,padding),                      # Output : 32 x 32
            *conv_block(64,64,kernel_size,stride,padding),                      # Output : 64 x 64
            nn.ConvTranspose2d(64,1,kernel_size,stride=stride,padding=padding)  # Output : 128 x 128
        )

        self.linear_decoder = nn.Sequential(
            nn.Linear(4096,2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048,2048),
            nn.LeakyReLU(0.2),
        )

        self.angle_decoder = nn.Sequential(
            nn.Linear(2048,config['resolution']**2),
            nn.ReLU()
        )
        '''self.gain_decoder = nn.Sequential(
            nn.Linear(2048,1),
            nn.Sigmoid(),
            )'''
        

    def forward(self, params):
        weights = self.model(params)
        weights = weights.view(params.shape[0],4096)
        weights = self.bottleneck(weights)

        # If decoder is convolutional
        #weights = weights.view(params.shape[0],512,2,2)
        #weights = self.decoder(weights)
        #weights = torch.exp(weights)

        # if Decoder is a linear layer
        weights = self.linear_decoder(weights)
        angles = self.angle_decoder(weights)
        
        #gain = self.gain_decoder(weights)*0.01
        #gain = torch.log(gain+1e-8)
        
        #weights = torch.exp(weights)
        angles = angles.view(params.shape[0],1,self.config['resolution'],self.config['resolution'])
        
        angles = angles[:,0,:,:,None,None,None,None,None]
        
        return [angles,params,None,None,None]

    def loss_function(self,
                      *args,
                      **kwargs):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        #tx_weights,rx_weights = args[0]
        combi_weights = args[0]
        input = args[1]
        recons = args[4]

        # NMSE Loss
        gt_norm = torch.sum(((input.flatten(1))**2),1)
        m_loss = torch.sum((input.flatten(1).cuda()-recons.flatten(1))**2,1)/(gt_norm.cuda())
        mse_loss = torch.mean(m_loss)

        # L1 Norm Regularizer
        #l1_norm = torch.mean(torch.norm(combi_weights.flatten(1),1,1))
        #l1_norm_loss = kwargs['l1_reg']*torch.mean(l1_norm)

        # L11 Norm Regularizer for group sparsity
        l1_norm = torch.mean(torch.norm(torch.norm(combi_weights.flatten(2),1,1),1,1))
        l1_norm_loss = kwargs['l1_reg']*torch.mean(l1_norm)
        
        
        # Overall Loss
        loss = mse_loss + l1_norm_loss
        
        return {'loss': loss, 'NMSE': mse_loss.detach(), 'KLD':mse_loss.detach()*0.0, 'L1 Norm':l1_norm.detach()}


    def get_trace_norm(self,channel):

        abs_channel = torch.abs(channel[:,0,:,:]+1j*channel[:,1,:,:])
        _,trace,_ = torch.svd(abs_channel)
        #trace_norm = torch.norm(abs_channel,p='nuc',dim=[1,2])
        trace_norm = torch.sum(torch.mean(trace,0)[1:])
        return trace_norm, abs_channel




class NN_Simulator(nn.Module):
    def __init__(self):
        super(NN_Simulator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.1))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        def conv_block(in_feat,out_feat,k_size,stride,padding=0):
            layers = [nn.ConvTranspose2d(in_feat,out_feat,k_size,stride=stride,padding=padding)]
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
            
        self.model = nn.Sequential(
            *block(3, 256),
            *block(256, 512),
            *block(512, 1024),
            *block(1024, 2048),
            *block(2048, 4096),
            *block(4096, 8192),
            nn.Linear(8192, 8192)
        )

        self.bottleneck_shape = (2048,2,2)

        self.conv_model = nn.Sequential(
            *conv_block(2048,512,2,2,0),
            *conv_block(512,256,3,1,1),
            *conv_block(256,128,3,1,1),
            *conv_block(128,64,3,1,1),
            nn.ConvTranspose2d(64,2,3,stride=1,padding=1),
        )
        #self.conv1 = nn.ConvTranspose2d(2048, 512, 2, stride=2)
        #self.conv2 = nn.ConvTranspose2d(512, 256, 3, stride=1, padding = 1)
        #self.conv3 = nn.ConvTranspose2d(256, 128, 3, stride=1, padding = 1)
        #self.conv4 = nn.ConvTranspose2d(128, 64, 3, stride=1, padding = 1)
        #self.conv5 = nn.ConvTranspose2d(64, 1, 3, stride=1, padding = 1)
        self.op_size = (2,4,4)
        

    def forward(self, params):
        img = self.model(params)
        #img = img.view(img.shape[0],*self.op_size)
        
        img = img.view(img.shape[0], *self.bottleneck_shape)
        img = self.conv_model(img)
        
        return img


    def get_trace_norm(self,channel):

        abs_channel = torch.abs(channel[:,0,:,:]+1j*channel[:,1,:,:])
        _,trace,_ = torch.svd(abs_channel)
        #trace_norm = torch.norm(abs_channel,p='nuc',dim=[1,2])
        trace_norm = torch.sum(torch.mean(trace,0)[1:])
        return trace_norm, abs_channel



