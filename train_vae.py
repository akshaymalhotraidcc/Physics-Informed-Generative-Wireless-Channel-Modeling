# Copyright (c) 2010-2025, InterDigital
# All rights reserved. 
# See LICENSE under the root folder.

'''
Use this script to train a Variational Autoencoder on saved channel data with an inline channel model.
'''

import argparse
import os
import numpy as np
import math
import sys
import pickle

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
from simulator import Model
import matplotlib.pyplot as plt
import datetime
import argparse
import yaml
from prettytable import PrettyTable
import vae_model as vae
import sim_model as auto
from torchinfo import summary
from scipy import sparse
from scipy.spatial import distance_matrix
from scipy.optimize import milp, LinearConstraint

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',    type = str,   default="test",  help ='The dataset to be used for training')
parser.add_argument('--batch_size', type = int,   default=128,                    help ='Batch size for training')
parser.add_argument('--lr',         type = float, default=2e-5,                   help ='Learning rate for training')
parser.add_argument('--num_epochs', type = int,   default=2000,                   help ='Number of epochs run over the training dataset')
parser.add_argument('--latent_dim', type = int,   default=64,                    help ='Latent dimension of VAE')
parser.add_argument('--l1_reg',  type = float,    default=0.0005,                   help ='Scaling Factor for Sparsity term')
parser.add_argument('--kld_reg',  type = float,   default=0.0005,                  help ='Scaling Factor for KL divergence term')
parser.add_argument('--resolution', type = int,   default=32,                     help ='Channel Resolution')
parser.add_argument('--relu_after', type = int,   default=5100,                   help ='ReLU Kicks in after this many epochs (Archived)')
parser.add_argument('--note',    type = str,      default=" ",                    help ='Optional Note to self')
parser.add_argument('--mode',    type = str,     default="vae",                   help ='Training Mode (Dont change)')
parser.add_argument('--load_dict',    type = bool,     default=False,             help ='Load saved dictionary')

# Generate experiment metadata
args = parser.parse_args()
config = vars(args)
print(config)
torch.autograd.set_detect_anomaly(True)

set_name = args.dataset

now = datetime.datetime.now()
now = now.strftime("%m%d_%H%M%S_")

exp_name = now+config['mode']
    
os.mkdir('weighted_model_results/'+exp_name)
with open('weighted_model_results/'+exp_name+'/config.yml', 'w') as outfile:
    yaml.dump(config, outfile, default_flow_style=False)

# Functions for performance metric calculations

def mmd_linear(gen_features, ref_features):
    '''
    MMD using a Linear Kernel

    gen_features : Array of size Num Samples x Features for Distribution 1
    ref_features : Array of size Num Samples x Features for Distribution 2
    '''
    
    gen_sim = gen_features@(np.transpose(gen_features))
    ref_sim = ref_features@(np.transpose(ref_features))
    cross_sim = gen_features@(np.transpose(ref_features))
    return np.mean(gen_sim) + np.mean(ref_sim) - 2 * np.mean(cross_sim)



# Dataset to hold and serve DFT channel samples

class Simulator_Dataset(Dataset):

    def __init__(self,dataset_name,mode='training'):

        with open('channel_data/'+dataset_name+'/'+dataset_name+'_labelled.pkl', 'rb') as f:
            self.data = pickle.load(f)

        if(mode=='training'):
            self.channels = self.data['training_dft']
            self.params = self.data['training_params']
        if(mode=='validation'):
            self.channels = self.data['val_dft']
            self.params = self.data['val_params']

    def __len__(self):
        return len(self.channels)

    def __getitem__(self,idx):
        channel = self.channels[idx].cuda()
        # Return only [ gain, AoA, Aod ] as params
        params = self.params[idx,:3]

        return channel,params

# Initialization for weights of a model (Optional)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

# Load dataset configuration file

with open('channel_data/'+set_name+'/config.yml', 'r') as stream:
    dataset_config = yaml.safe_load(stream)

# Hyperparameters
        
num_paths = dataset_config['num_paths']
lr = args.lr
b1 = 0.5
b2 = 0.999

batch_size = args.batch_size
n_epochs   = args.num_epochs

num_ants = dataset_config['num_ants']
img_shape = (batch_size,2,num_ants[0],num_ants[0])
param_shape = (batch_size,3)
Tensor = torch.cuda.FloatTensor
reg_lambda = config['l1_reg']

# Gain matrix prediction model

wm = vae.WeightsVAE(in_channels=2,latent_dim=args.latent_dim, config=config, use_conv=False)    
result_vae = summary(wm, input_size=img_shape)

with open('weighted_model_results/'+exp_name+'/vae_summary.txt', 'w') as text_file:
    text_file.write(str(result_vae))

# Physics based Channel Model

model = Model(1,[num_ants[0],1],[num_ants[0],1])

# Configure data loader

tr_dataset = Simulator_Dataset(set_name,mode='training')
tr_dataloader = DataLoader(tr_dataset,batch_size=batch_size,shuffle=True)

val_dataset = Simulator_Dataset(set_name,mode='validation')
val_dataloader = DataLoader(val_dataset,batch_size=batch_size,shuffle=True)

# Initialize Array Response Dictionaries and weights

resolution = args.resolution
combi_responses = torch.zeros(1,resolution,resolution,1,num_ants[0],num_ants[0],dtype=torch.complex64).cuda()

if(config['load_dict']):
    scenario = '_'.join(dataset_config['datafolder'].split('_')[:-1])
    print(scenario)
    combi_responses = torch.load('utilities/response_dict_'+str(dataset_config['num_ants'][0])+'ants_'+str(config['resolution'])+'r_'+scenario)
    lim = 120*(np.pi/180)
else:
    lim = 1.5708 # Upper/Lower limit of Array response dictionaries

    for aoa_scale in range(resolution):
        for aod_scale in range(resolution):
            aoa = torch.tensor([(((aoa_scale+1)/resolution)*2.0 - 1.0)*lim]).reshape(1,1).cuda()
            zoa = torch.tensor([1.5708]).reshape(1,1).cuda()
            rxr = model.txAnt.getResponse(aoa,zoa)[:,:,None,:,None]
        
            aod = torch.tensor([(((aod_scale+1)/resolution)*2.0 - 1.0)*lim]).reshape(1,1).cuda()    
            zod = torch.tensor([1.5708]).reshape(1,1).cuda()
            txr = model.rxAnt.getResponse(aod,zod)[:,None,:,:,None]

            combi_r = (rxr*txr)[:,:,:,0,0]
            combi_responses[0,aoa_scale,aod_scale] = combi_r

# Optimizers

optimizer = torch.optim.Adam(wm.parameters(), lr=lr, betas=(b1, b2))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)
criterion = nn.MSELoss()
val_criterion = nn.MSELoss()

# Training

angles = []
gains = []
losses = []
val_losses = []

negative_weights = []
mse_losses = []
l1_norms = []
klds = []

w_dists = []
mmds = []

for epoch in range(n_epochs):

    '''
    Train VAE
    '''
    
    for i, (ref_channel,params) in enumerate(tr_dataloader):

        optimizer.zero_grad()
        ones = torch.ones(params.shape[0]*num_paths)

        if(epoch>config['relu_after']):
            force_relu=True
        else:
            force_relu = False

        # Produce predicted gain matrix
        results = wm.forward(ref_channel.cuda(),force_relu=force_relu)
        combi_weights, input, mu, log_var,_ = results

        # Generate channel using predicted gain matrix
        channel = model.getChannelfromCombiArrayResponse(None, combi_weights,combi_responses,use_gains=False,num_paths=dataset_config['num_paths'])
        channel = torch.stack(model.get_dft(channel))
        channel = torch.stack([torch.real(channel),torch.imag(channel)]).permute(1,0,2,3)
        
        # Create input for loss function
        results[1] = ref_channel
        results[4] = channel
        loss = wm.loss_function(*results,M_N = config['kld_reg'],l1_reg=args.l1_reg)

        loss['loss'].backward()
        optimizer.step()
    
        if(i%10==0):
            plt.imshow(combi_weights[5,:,:,0,0,0].cpu().detach().numpy(),origin='lower')
            plt.colorbar(location='bottom')
            plt.tight_layout()
            plt.savefig('weighted_model_results/'+exp_name+'/tr_weights')
            plt.clf()
            plt.close()    
            print(np.round(loss['NMSE'].item(),5),'|',np.round(loss['L1 Norm'].item(),3),'|',np.round(loss['KLD'].item(),3))

    # Save model logs and graphs
    mse_losses.append(loss['NMSE'].item())
    l1_norms.append(loss['L1 Norm'].item())
    klds.append(loss['KLD'].item())

    plt.figure(figsize=(6,8))
    
    plt.subplot(3,1,1)
    plt.plot(mse_losses,label='NMSE Loss')
    plt.semilogy()
    plt.grid('on')
    plt.ylabel('NMSE Loss')

    plt.subplot(3,1,2)
    plt.plot(l1_norms,label='L1 Norm')
    plt.grid('on')
    plt.semilogy()
    plt.ylabel('L1 Norm')

    plt.subplot(3,1,3)
    plt.plot(klds,label='KLD')
    plt.grid('on')
    plt.semilogy()
    plt.ylabel('KL Divergence')
    plt.xlabel('Epoch')

    plt.tight_layout()
    plt.savefig('weighted_model_results/'+exp_name+'/losses')
    plt.clf()
    plt.close()

    # Validate
    
    val_gains = []
    for i, (ref_channel,params) in enumerate(val_dataloader):

        scale = resolution/2
        results = wm.forward(ref_channel.cuda(),force_relu=force_relu)
        combi_weights = results[0]

        '''
        Generate sample outputs of the prediction mechanism (Not generation)
        '''
        plt.figure(figsize=(30,20))
        for bdx in range(3):
            ax = plt.subplot(2,3,bdx+1)
            plt.imshow(np.abs(combi_weights[bdx,:,:,0,0,0].cpu().detach().numpy()),vmin=0.0,origin='lower')
            plt.colorbar(location='bottom')
            
            for i in range(num_paths):
                plt.scatter(0,0,marker='*',s=1, color='black', edgecolor='black', linewidth=1, label=str(np.round(params[bdx,2,i].cpu().numpy(),3))+' / '+str(np.round(params[bdx,1,i].cpu().numpy(),3)))
                pass
            
            plt.xticks([i for i in range(resolution)],[np.round(lim*((i-scale)/scale),2) for i in range(resolution)],rotation=45)
            plt.yticks([i for i in range(resolution)],[np.round(lim*((i-scale)/scale),2) for i in range(resolution)])

            plt.legend(title='AoD / AoA')
            plt.xlabel('AoD')
            plt.ylabel('AoA')

            plt.subplot(2,3,bdx+4)
            plt.imshow(combi_weights[bdx,:,:,0,0,0].cpu().detach().numpy(),origin='lower')
            plt.colorbar(location='bottom')    
            
            plt.xticks([i for i in range(resolution)],[np.round(lim*((i-scale)/scale),2) for i in range(resolution)],rotation=45)
            plt.yticks([i for i in range(resolution)],[np.round(lim*((i-scale)/scale),2) for i in range(resolution)])

            plt.xlabel('AoD')
            plt.ylabel('AoA')

        plt.tight_layout()
        plt.savefig('weighted_model_results/'+exp_name+'/selected_weights')
        plt.clf()
        plt.close()
        break

    print('Avg. Max Gain: ',np.mean(val_gains))
    gains.append(np.mean(val_gains))

    '''
    Generate new channels from random noise
    '''
    
    # Sample noise as generator input
    z = torch.randn(batch_size,config['latent_dim']).cuda()

    # Generate gain prediction matrices and produce channel
    combi_weights = wm.decode(z,force_relu=False)
    fake_channel = model.getChannelfromCombiArrayResponse(None, combi_weights,combi_responses,use_gains=False,num_paths=dataset_config['num_paths'])
    fake_channel = torch.stack(model.get_dft(fake_channel))
    fake_channel = torch.stack([torch.real(fake_channel),torch.imag(fake_channel)]).permute(1,0,2,3)

    fake_channel = fake_channel.cpu().detach()
    ref_channel = ref_channel.cpu().detach()
    fake_ch_flat = fake_channel.flatten(1)
    ref_ch_flat = ref_channel.flatten(1)

    # Calculate distances between real distribuion and generated distribution.
    wasserstein_distance = wasserstein_dist(fake_ch_flat,ref_ch_flat)
    mmd = mmd_linear(fake_ch_flat,ref_ch_flat)
    print('Wasserstein Distance :',wasserstein_distance)
    print('MMD :',mmd)
    print()

    # Save loss curves and metrics
    w_dists.append(wasserstein_distance)
    mmds.append(mmd)

    plt.plot(gains)
    plt.grid()
    plt.tight_layout()
    plt.savefig('weighted_model_results/'+exp_name+'/gains')
    plt.clf()
    plt.close()

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(w_dists,label='Wasserstein Distance')
    plt.grid('on')
    #plt.semilogy()
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(mmds,label='Maximum Mean Discrepancy')
    plt.grid('on')
    #plt.semilogy()
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('weighted_model_results/'+exp_name+'/distances')
    plt.clf()
    plt.close()

    # Save models.
    np.save('weighted_model_results/'+exp_name+'/mmds.npy',mmds)
    np.save('weighted_model_results/'+exp_name+'/wasserstein_distances.npy',w_dists)
    torch.save(wm.state_dict(),'weighted_model_results/'+exp_name+'/weighted_array_model.pt')

        
            