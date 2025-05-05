# Copyright (c) 2010-2025, InterDigital
# All rights reserved. 
# See LICENSE under the root folder.

import os
import numpy as np
import scipy
from scipy.interpolate import interp1d
import DeepMIMOv3
import matplotlib.pyplot as plt
import pickle
import argparse
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--scenario',    type = str,   default="asu_campus1",                 help ='Scenario')
parser.add_argument('--paths',    type = int,   default=2,                 help ='Scenario')
parser.add_argument('--a_start',    type = int,   default=0,                 help ='Scenario')
parser.add_argument('--a_end',      type = int,   default=10000,                 help ='Scenario')

# For Dataset Generation
parser.add_argument('--dataset',    type = str,   default="test",           help ='The name of the dataset to save')
parser.add_argument("--deepmimo", action="store_true",                      help="Use DeepMIMO Scenarios") 
parser.add_argument('--num_ants',   nargs='+',     type = int, default=[16,1], help ='Number of Antennas')

parser.add_argument('--pow', nargs='+',     type = float, default=[0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001], help ='Upper/Lower Limits of Tx Power per Path')

parser.add_argument('--phase', nargs='+',   type = float, default=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],                                    help ='Upper/Lower Limits of Tx Phase per Path (Unused)')

parser.add_argument('--tau', nargs='+',     type = float, default=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],                                    help ='Upper/Lower Limits of Tx Delay per Path (Unused)')

parser.add_argument('--unif_min_scale',  type = float,   default=-1.0,                   help ='Scaling to upper limit to get lower limit')
parser.add_argument('--resolution',      type = int,   default=16,                       help ='Resolution for On-Grid Data Only')
parser.add_argument('--zresolution',      type = int,   default=16,                       help ='Resolution for On-Grid Data Only')
parser.add_argument('--bs',      type = int,   default=1,                       help ='Resolution for On-Grid Data Only')

# Generate experiment metadata
args = parser.parse_args()
args.deepmimo = True

print(args.num_ants[0])

# Load the default parameters
parameters = DeepMIMOv3.default_params()

# Set scenario name
parameters['scenario'] = args.scenario

# Set the main folder containing extracted scenarios
parameters['dataset_folder'] = "deepmimo_data/"

parameters['num_paths'] = 100
parameters['active_BS'] = [args.bs]
parameters['bs_antenna']['shape'] = np.array([4, 4])
parameters['bs_antenna']['spacing'] = 0.5
parameters['bs_antenna']['FoV'] = np.array([240, 180])
parameters['bs_antenna']['radiation_pattern'] = 'isotropic'
parameters['ue_antenna']['shape'] = np.array([4, 4])
parameters['ue_antenna']['spacing'] = 0.5
parameters['ue_antenna']['FoV'] = np.array([240, 180])
parameters['ue_antenna']['radiation_pattern'] = 'isotropic'
parameters['enable_doppler'] = 0
parameters['enable_dual_polar'] = 0
parameters['enable_BS2BS'] = 0
parameters['OFDM_channels'] = 1 # Frequency (OFDM) or time domain channels
parameters['OFDM']['subcarriers'] = 512
parameters['OFDM']['selected_subcarriers'] = np.array([0])
parameters['OFDM']['bandwidth'] = 0.05
parameters['OFDM']['RX_filter'] = 0

parameters['user_rows'] = np.arange(args.a_start,args.a_end)


if(True):
    # Generate data
    dataset = DeepMIMOv3.generate_data(parameters)

    # Cluster Paths
    thresh_num_paths = 1
    dist_threshold = 1

    clustered_paths = []

    for i in range(len(dataset[0]['user']['paths'])):

        bins = []
        
        if(dataset[0]['user']['paths'][i]['num_paths']>(thresh_num_paths)):
            params = np.stack([dataset[0]['user']['paths'][i]['power'],
                               dataset[0]['user']['paths'][i]['DoA_phi'],
                               dataset[0]['user']['paths'][i]['DoD_phi'],
                               dataset[0]['user']['paths'][i]['DoA_theta'],
                               dataset[0]['user']['paths'][i]['DoD_theta']]).T

            # Go through each path
            for pdx in range(len(params)):
                # If bins are empty, create a new bin
                if(pdx==0):
                    bins.append(np.array([params[pdx]]))

                else:

                    # Flag to create a new bin for the path
                    new_bin = True
                    
                    # Go through each bin
                    for bdx in range(len(bins)):

                        # Calculate the average distance from all points in the bin
                        dist = np.sqrt(np.sum((bins[bdx][:,1:]-params[pdx,1:])**2))

                        # If the path is closer on average to all the points in the bin
                        if(dist<dist_threshold):
                            
                            # Do not make a new bin for the path
                            new_bin = False
                            # Add the current path to the bin
                            bins[bdx] = np.vstack((bins[bdx],params[pdx]))

                    # If path is not close to any of the existing bins
                    if(new_bin):
                        
                        # Create a new bin
                        bins.append(np.array([params[pdx]]))

            # Add bins to the overall set
            clustered_paths.append(bins)

    #for idx in range(1,20):
    #    print('For',idx,'paths :',len(bins[idx]),'samples available...')

    all_paths = []
    # For each set of bins for every point
    for bins in clustered_paths:
        path = []
        
        # For each bin in the set
        for b in bins:
            
            # Use the average parameters in every bin
            #path.append(np.mean(b,0))

            # Use the path with highest gain in every bin
            path.append(b[np.argmax(b[:,0]),:])

        path = np.stack(path)
        all_paths.append(path)

    num_paths = args.paths

    paths = []
    # Select only those paths with more than the required paths
    for p in all_paths:
        if(len(p)>(num_paths-1)):
            paths.append(p[:num_paths])

    paths = np.stack(paths)

    print()
    print('!! For',num_paths,' paths there are',paths.shape[0],'potential samples available !!')
    print()

    plt.figure(figsize=(12,4*num_paths))
    for pdx in range(num_paths):
        #for idx in range(4):
        plt.subplot(num_paths,4,1+4*pdx)
        plt.scatter(paths[:,pdx,1],paths[:,pdx,2],alpha=0.01)
        plt.grid()
        
        plt.subplot(num_paths,4,2+4*pdx)
        plt.scatter(paths[:,pdx,3],paths[:,pdx,4],alpha=0.01)
        plt.grid()

    print("Path distributions saved in :images/path_plot")
    plt.savefig('images/path_plot')

    zoa_lim   = -180
    zod_lim   = -180
    aoa_lim = -90
    aod_lim = -90

    def get_cond_elevation(entry,aoa_min=0.0,aod_min=0.0):

        #c1 = np.all(entry['DoA_theta']>aoa_min) and np.all(entry['DoA_theta']<60+aoa_min)
        #c2 = np.all(entry['DoD_theta']>aod_min) and np.all(entry['DoD_theta']<60+aod_min)
        
        c1 = np.all(entry[3]>aoa_min) and np.all(entry[3]<360+aoa_min)
        c2 = np.all(entry[4]>aod_min) and np.all(entry[4]<360+aod_min)

        c = c1 and c2
        return c

    def get_cond_azimuth(entry,aoa_min=0.0,aod_min=0.0):

        #c1 = np.all(entry['DoA_phi']>aoa_min) and np.all(entry['DoA_phi']<aoa_min+120)
        #c2 = np.all(entry['DoD_phi']>aod_min) and np.all(entry['DoD_phi']<aod_min+120)

        c1 = np.all(entry[1]>aoa_min) and np.all(entry[1]<aoa_min+180)
        c2 = np.all(entry[2]>aod_min) and np.all(entry[2]<aod_min+180)

        c = c1 and c2
        return c

    parameters['elevation_lim'] = [zoa_lim,zoa_lim+360]
    parameters['azimuth_arr_lim'] = [aoa_lim,aoa_lim+180]
    parameters['azimuth_dep_lim'] = [aod_lim,aod_lim+180]

    print('Filtering...')
    print('Elevation Limits :',parameters['elevation_lim'])
    print('Azimuth Arrival Limits :',parameters['azimuth_arr_lim'])
    print('Azimuth Departure Limits :',parameters['azimuth_dep_lim'])
    ds = []

    
    for i in range(len(paths)):
        if(get_cond_elevation(paths[i].T,parameters['elevation_lim'][0],parameters['elevation_lim'][0]) and 
           get_cond_azimuth(paths[i].T,parameters['azimuth_arr_lim'][0],parameters['azimuth_dep_lim'][0])):
                paths[i][:,1:] = paths[i][:,1:]*(np.pi/180)
                ds.append(paths[i])

    ds = np.stack(ds).transpose(0,2,1)
    print()
    print('!! After filtering, there are',ds.shape[0],'samples available. !!')
    print()
    try:
        del parameters['user_rows']
    except:
        pass

    try:
        os.mkdir('city_scenarios/param_datasets/'+parameters['scenario']+'_'+str(num_paths)+'p')
    except:
        pass
        
    with open('city_scenarios/param_datasets/'+parameters['scenario']+'_'+str(num_paths)+'p/config', 'wb') as f:
        pickle.dump(parameters, f)
        
    np.save('city_scenarios/param_datasets/'+parameters['scenario']+'_'+str(num_paths)+'p/params.npy',ds)

    plt.figure(figsize=(6,(num_paths*2+2)))
    for p in range(num_paths):
        ax = plt.subplot(num_paths,2,2*p+1)
        plt.scatter(ds[:,1,p],ds[:,2,p],alpha=0.1)
        plt.grid()
        plt.xlabel('AoA')
        plt.ylabel('AoD')
        ax.set_title('Path '+str(p)+' Azimuth')
        ax = plt.subplot(num_paths,2,2*p+2)
        plt.scatter(ds[:,3,p],ds[:,4,p],alpha=0.1)
        plt.xlabel('ZoA')
        plt.ylabel('ZoD')
        ax.set_title('Path '+str(p)+' Elevation')
        plt.grid()

    plt.tight_layout()
    plt.savefig('city_scenarios/param_datasets/'+parameters['scenario']+'_'+str(num_paths)+'p/paths',bbox_inches='tight')
    plt.show()

    titles = ['AoA','AoD','ZoA','ZoD']

    plt.figure(figsize=(10,10))
    for i in range(1,5):
        ax = plt.subplot(2,2,i)
        plt.hist(ds[:,i].flatten()*(180/np.pi))
        ax.set_title(titles[i-1])
        plt.grid()

    plt.tight_layout()
    plt.savefig('city_scenarios/param_datasets/'+parameters['scenario']+'_'+str(num_paths)+'p/param_dists',bbox_inches='tight')


create_dataset = False
check = input('Create Dataset for Scenario? : ')
if(check=='y'):
    print('Creating Dataset for Scenario...')
    create_dataset = True
else:
    create_dataset = False

if(create_dataset):
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
        
    # Create Dataset configuration file
    config = vars(args)
    config['num_paths'] = config['paths']
    config['datafolder'] = config['scenario']+'_'+str(config['paths'])+'p'
    config['pow'] = config['pow'][:2*(config['paths'])]
    config['tau'] = config['tau'][:2*(config['paths'])]
    
    print(config)
    
    with open('channel_data/'+args.dataset+'/config.yml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)
    
    # Generate DFT channel dataset
    
    m = Model(numPaths=config['paths'],nt=[config['num_ants'][0],1],nr=[config['num_ants'][0],1])

    t_config = copy.deepcopy(config)
    t_config['num_ants'] = args.num_ants[0]
    ds = m.save_data(500,args.dataset,t_config)
    # Generate and save dataset statistics plots
    
    plt.figure(figsize=(14,6))
    
    plt.subplot(1,2,1)
    for n in range(config['paths']):
        plt.scatter(ds['training_params'][:,2,n],ds['training_params'][:,1,n],label='Path '+str(n))
    
    plt.xticks([(1.0*(2*i-10))/10 for i in range(11)],rotation=45,fontsize=12)
    plt.yticks([(1.0*(2*i-10))/10 for i in range(11)],fontsize=12)
    plt.xlabel(r"$\theta^d_p$",fontsize=18,labelpad=-8)
    plt.ylabel(r"$\theta^a_p$",fontsize=18,labelpad=-8)
    plt.grid()
    
    plt.subplot(1,2,2)
    
    for n in range(config['paths']):
        plt.scatter(ds['training_params'][:,4,n],ds['training_params'][:,3,n],label='Path '+str(n))
    
    plt.xticks([1.57+(0.5*(2*i-10))/10 for i in range(11)],rotation=45,fontsize=12)
    plt.yticks([1.57+(0.5*(2*i-10))/10 for i in range(11)],fontsize=12)
    plt.xlabel(r"$\phi^d_p$",fontsize=18,labelpad=-8)
    plt.ylabel(r"$\phi^a_p$",fontsize=18,labelpad=-8)
    plt.legend()
    plt.grid()
    
    #plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig('channel_data/'+args.dataset+'/parameter_plot.png')
    #plt.savefig('images/parameter_plot.png')
        
    fig = plt.figure(figsize=(20,4*args.paths))
    plt_idx = 1
    for key in m.stats.keys():
    
        if('placeholder' in key):
            pass
        else:
            ax = plt.subplot(args.paths,6,plt_idx)
            ax.set_title(key)
            plt.hist(m.stats[key],bins=20)
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

print()

create_dict = False
check = input('Create Dictionary for Scenario? : ')
if(check=='y'):
    print('Creating Dictionary for Scenario...')
    create_dict = True
else:
    create_dict = False

if(create_dict):
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
    import vae_model as m
    from torchinfo import summary
    from scipy import ndimage
    from skimage.feature import peak_local_max
    import time
    
    import torch
    import numpy as np
    from simulator import Model
    
    scenario = parameters['scenario']
    resolution = args.resolution
    zresolution = 16
    num_ants = args.num_ants
        
    with open('city_scenarios/param_datasets/'+scenario+'_'+str(args.paths)+'p/config', 'rb') as f:
        config = pickle.load(f)
    
    print('Z limits',config['elevation_lim'])
    print('AoA limits',config['azimuth_arr_lim'])
    print('AoD limits',config['azimuth_dep_lim'])
    
    arr_center = config['azimuth_arr_lim'][0] + (config['azimuth_arr_lim'][1]-config['azimuth_arr_lim'][0])/2
    dep_center = config['azimuth_dep_lim'][0] + (config['azimuth_dep_lim'][1]-config['azimuth_dep_lim'][0])/2

    print('Azimuth arrival center :',arr_center)
    print('Azimuth departure center :',dep_center)
    
    aoa_bias = arr_center*(np.pi/180)
    aod_bias = dep_center*(np.pi/180)
    
    model = Model(1,num_ants,num_ants)

    print('Constructing dictionary...')
    # Initialize Array Response Dictionaries and weights
    rx_arr = []
    tx_arr = []
    
    aoas = []
    aods = []
    
    combi_responses = torch.zeros(1,resolution,resolution,1,num_ants[0],num_ants[0],1,1,dtype=torch.complex64).cuda()
    combi_map = torch.zeros(resolution,resolution,2)
    
    s = time.time()
    
    aoa_lim = 120*(np.pi/180)
    aod_lim = 120*(np.pi/180)
    zlim = 30*(np.pi/180)
    
    
    for aoa_scale in range(resolution):
        for aod_scale in range(resolution):
            aoa = torch.tensor([aoa_bias+(((aoa_scale+1)/resolution)*2.0 - 1.0)*aoa_lim]).reshape(1,1).cuda()
            zoa = torch.tensor([1.5708]).reshape(1,1).cuda()
            rxr = model.rxAnt.getResponse(aoa,zoa)[:,:,None,:,None]
        
            aod = torch.tensor([aod_bias+(((aod_scale+1)/resolution)*2.0 - 1.0)*aod_lim]).reshape(1,1).cuda()    
            zod = torch.tensor([1.5708]).reshape(1,1).cuda()
            txr = model.txAnt.getResponse(aod,zod)[:,None,:,:,None]

            combi_r = rxr*txr
            combi_responses[0,aoa_scale,aod_scale] = combi_r
            combi_map[aoa_scale,aod_scale] = torch.tensor([aoa,aod])
    
    e = time.time()
    print('Time to create dictionary:',e-s,'seconds')
    print('Dictionary Shape:',combi_responses.shape)
    print('Dictionary size:',(torch.numel(combi_responses)*64)/(8*1024*1024),'Mb')
    torch.save(combi_responses,'utilities/response_dict_'+str(num_ants[0])+'ants_'+str(resolution)+'r_'+scenario)
    torch.save(combi_map,'utilities/map_'+str(num_ants[0])+'ants_'+str(resolution)+'p_'+scenario)