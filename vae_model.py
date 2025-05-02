import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
import numpy as np
from simulator import Model
import sim_model as m
from utils import barrier_function, gen_sigmoid, relu_barrier
from torch.autograd import Variable

class WeightsVAE(nn.Module):
    
    def __init__(self,
                 in_channels,
                 latent_dim,
                 config,
                 use_conv = False,
                 hidden_dims = None,
                 **kwargs) -> None:
        super(WeightsVAE, self).__init__()

        self.latent_dim = latent_dim
        self.use_conv = use_conv

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

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
        
        self.encoder = nn.Sequential(*modules)

        # Input should be 4608 for larger antennas
        self.lin1 = nn.Linear(2048,2048)
        self.relu1 = nn.LeakyReLU(0.2)
        self.lin2 = nn.Linear(2048,1024)
        self.relu2 = nn.LeakyReLU(0.2)
        self.lin3 = nn.Linear(1024,512)
        self.relu3 = nn.LeakyReLU(0.2)
        
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_var = nn.Linear(512, latent_dim)

        
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
        hidden_dims.reverse()

        def block(in_feat, out_feat):
            layers = [nn.Linear(in_feat, out_feat)]
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.decoder_hidden_dims = [512,256,128,64,32,16]

        self.decoder_prep = nn.Sequential(
            nn.Linear(latent_dim,256),
            nn.LeakyReLU(0.2),
            nn.Linear(256,1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024,2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048,4096)
        )

        def conv_block(in_feat,out_feat,k_size,stride,padding=0,dilation=1):
            
            layers = [nn.ConvTranspose2d(in_feat,out_feat,k_size,stride=stride,padding=padding,dilation=dilation,output_padding=1)]
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            #layers.append(nn.ConvTranspose2d(in_feat,out_feat,k_size,stride=stride,padding=padding,dilation=dilation))
            #layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        def linear_sequence(in_feat,out_feat,num_blocks=1):
            layers = []
            for n in range(num_blocks):
                layers.append(nn.Linear(in_feat,out_feat))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        def conv_layer(in_feat,out_feat,k_size,stride,padding=0,dilation=1):
            
            layers = [nn.ConvTranspose2d(in_feat,out_feat,k_size,stride=stride,padding=padding,dilation=dilation,)]
            return layers

        stride   = 2
        padding  = 1
        dilation = 1
        kernel_size = 3

        if(use_conv):
            # Purely convolutional decoder
            self.decoder = nn.Sequential(
                *conv_block(256,256,kernel_size,stride,padding,dilation),  # 8 / 8
                *conv_block(256,128,kernel_size,stride,padding,dilation),  # 12 / 16
                *conv_block(128,128,kernel_size,stride,padding,dilation),   # 16 / 32
                *conv_block(128,128,kernel_size,stride,padding,dilation),    # 20 / 64
                *conv_layer(128,1,1,1,0,1),
            )
            
        else:
            # Linear Decoder
            self.decoder = nn.Sequential(
                *linear_sequence(4096,1024),
                *linear_sequence(1024,1024),
                nn.Linear(1024,config['resolution']**2),
                #nn.Linear(128,6),
            )

        self.config = config

        # For high temp sigmoid
        self.temperature = 3.0
        self.final_sigmoid = nn.Sigmoid()

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        result = self.lin1(result)
        result = self.relu1(result)

        result = self.lin2(result)
        result = self.relu2(result)

        result = self.lin3(result)
        result = self.relu3(result)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z, force_relu):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """

        # For NN based simulator
        weights = self.decoder_prep(z)

        # Using Convolutional Decoder
        if(self.use_conv):
            weights = weights.reshape(z.shape[0],256,4,4)
            weights = self.decoder(weights)
           
        else:
            # Using Linear Decoder
            weights = self.decoder(weights)

            if(force_relu):
                weights = F.relu(weights) #torch.clip(weights,min=0.0,max=None)
            
            # Change the following    
            weights = weights.view(z.shape[0],1,self.config['resolution'],self.config['resolution'])

        weights = (weights[:,0,:,:,None,None,None])

        #weights = weights.view(z.shape[0],3,2)
        
        return weights

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, force_relu=False, **kwargs):

        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)

        weights = self.decode(z,force_relu)
        return  [weights, input, mu, log_var, None]

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

        combi_weights = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        recons = args[4]

        # NMSE Loss
        gt_norm = torch.sum(((input.flatten(1))**2),1)
        m_loss = torch.sum((input.flatten(1).cuda()-recons.flatten(1))**2,1)/(gt_norm.cuda())
        mse_loss = torch.mean(m_loss)

        # L11 Norm Regularizer for group sparsity
        
        # If Combi weights are in db
        #combi_weights = 10**((10*combi_weights)/10)
        
        l1_norm = torch.mean(torch.norm(torch.norm(combi_weights.flatten(2),1,1),1,1))
        l1_norm_loss = kwargs['l1_reg']*torch.mean(l1_norm)
        
        
        # KL Divergence Loss
        
        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        # Overall Loss
        loss = mse_loss + l1_norm_loss + kld_weight * kld_loss
        
        return {'loss': loss, 'NMSE': mse_loss.detach(), 'KLD':kld_loss.detach(), 'L1 Norm':l1_norm.detach()}

    def sample(self,
               num_samples, dft = False, **kwargs):
        print('Sampling...')
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,self.latent_dim).cuda()
        weights = self.decode(z)
        return weights