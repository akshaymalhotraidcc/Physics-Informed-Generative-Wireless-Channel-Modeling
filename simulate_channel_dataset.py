import argparse
import os
import numpy as np
import math
import sys

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
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',    type = str,   default="test",           help ='The name of the dataset to save')
parser.add_argument('--size',       type = int,   default=20000,            help ='The size of the dataset (Not for DeepMIMO)')
parser.add_argument("--on_grid", action="store_true",                       help="Generate Ongrid Data") 
parser.add_argument("--deepmimo", action="store_true",                      help="Use DeepMIMO Scenarios") 
parser.add_argument('--base_id',  type = int,   default=1,                  help ='Number of Paths to generate')
parser.add_argument('--datafolder',  type = str,   default='',              help ='Source folder of DeepMIMO data (Only DeepMIMO)')

parser.add_argument('--num_paths',  type = int,   default=5,                help ='Number of Paths to generate')
parser.add_argument('--num_ants',   type = int,   default=16,               help ='Number of Antennas')

parser.add_argument('--pow', nargs='+',     type = float, default=[0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001], help ='Upper/Lower Limits of Tx Power per Path')
parser.add_argument('--aoa', nargs='+',     type = float, default=[0.7,0.15,-0.55,0.3,0.6,0.15,0.1,0.2,-0.8,0.2],           help ='Upper/Lower Limits of Tx AoA per Path')
parser.add_argument('--aod', nargs='+',     type = float, default=[0.7,0.15,-0.55,0.3,-0.6,-0.15,0.6,0.2,0.3,0.2],          help ='Upper/Lower Limits of Tx AoD per Path')
parser.add_argument('--zoa', nargs='+',     type = float, default=[1.5708,1.5708,1.5708,1.5708,1.5708],                     help ='Upper/Lower Limits of Tx ZoA per Path')
parser.add_argument('--zod', nargs='+',     type = float, default=[1.5708,1.5708,1.5708,1.5708,1.5708],                     help ='Upper/Lower Limits of Tx ZoD per Path')
parser.add_argument('--phase', nargs='+',   type = float, default=[0.0,0.0,0.0,0.0,0.0],                                    help ='Upper/Lower Limits of Tx Phase per Path (Unused)')
parser.add_argument('--tau', nargs='+',     type = float, default=[0.0,0.0,0.0,0.0,0.0],                                    help ='Upper/Lower Limits of Tx Delay per Path (Unused)')

parser.add_argument('--unif_min_scale',  type = float,   default=-1.0,                   help ='Scaling to upper limit to get lower limit')
parser.add_argument('--resolution',      type = int,   default=32,                       help ='Resolution for On-Grid Data Only')

    
# Create Dataset configuration file

args = parser.parse_args()
config = vars(args)
config['pow'] = config['pow'][:2*(config['num_paths'])]
config['aoa'] = config['aoa'][:2*(config['num_paths'])]
config['aod'] = config['aod'][:2*(config['num_paths'])]
config['zoa'] = config['zoa'][:2*(config['num_paths'])]
config['zod'] = config['zod'][:2*(config['num_paths'])]
config['phase'] = config['phase'][:2*(config['num_paths'])]
config['tau'] = config['tau'][:2*(config['num_paths'])]

print(config)

# Create a new folder and overwrite if already existing

overwrite = False
if(os.path.isdir('channel_data/'+args.dataset)):
    check = input('Overwrite existing Dataset? : ')
    if(check=='y'):
        print('Overwriting...')
        overwrite = True
    else:
        overwrite = False
        args.dataset = input('Provide new Set Name : ')

if(not(overwrite)):
    os.mkdir('channel_data/'+args.dataset)
    
with open('channel_data/'+args.dataset+'/config.yml', 'w') as outfile:
    yaml.dump(config, outfile, default_flow_style=False)

# Generate DFT channel dataset

m = Model(numPaths=config['num_paths'],nt=[config['num_ants'],1],nr=[config['num_ants'],1])
if(args.size>500):
    ds = m.save_data(args.size//500,args.dataset,config)
else:
    ds = m.save_data(1,args.dataset,config,batch=args.size)

# Generate and save dataset statistics plots

plt.figure(figsize=(8,8))
for n in range(config['num_paths']):
    plt.scatter(ds['training_params'][:,2,n],ds['training_params'][:,1,n],label='Path '+str(n))

fsize = 20
plt.xticks([-1.0,-0.75,-0.5,-0.25,0.0,0.25,0.5,0.75,1.0],fontsize=fsize-2,rotation=45)
plt.yticks([-1.0,-0.75,-0.5,-0.25,0.0,0.25,0.5,0.75,1.0],fontsize=fsize-2)
plt.xlabel('AoD (Rad.)',fontsize=fsize)
plt.ylabel('AoA (Rad.)',fontsize=fsize)
plt.grid()
plt.legend(fontsize=fsize)
plt.tight_layout()
plt.savefig('channel_data/'+args.dataset+'/parameter_plot.png')
    
fig = plt.figure(figsize=(20,4*args.num_paths))
plt_idx = 1
for key in m.stats.keys():

    if('phase' in key):
        pass
    else:
        ax = plt.subplot(args.num_paths,6,plt_idx)
        ax.set_title(key)
        plt.hist(m.stats[key],bins=10)
        plt.grid()
    plt_idx+=1
plt.tight_layout()
plt.savefig('channel_data/'+args.dataset+'/parameter_distributions.png')
plt.clf()
plt.close()

# Generate and save sample channels

plt.figure(figsize=(8,8))
for k in range(12):
    plt.subplot(4,3,k+1)
    plt.axis('off')
    plt.imshow(ds['training_dft'][k].cpu().detach().numpy().reshape(ds['training_dft'][k].shape[0]*ds['training_dft'][k].shape[1],ds['training_dft'][k].shape[2]).T)

plt.tight_layout()
plt.savefig('channel_data/'+args.dataset+'/check_channels.png',bbox_inches='tight')
