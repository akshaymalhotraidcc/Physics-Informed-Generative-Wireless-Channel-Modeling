# Copyright (c) 2010-2025, InterDigital
# All rights reserved. 
# See LICENSE under the root folder.

import numpy as np
import time
import scipy
import torch
import yaml
import os
import scipy
import matplotlib.pyplot as plt
import pickle
import random
from opt_einsum import contract

class AntPanel:

    def __init__(self, ny, nz=1, antType='Isotropic', antSpacingOverLambda=.5, centered=False, order='zTopLeft'):
        print(ny,nz)
        self.ny, self.nz = int(ny), int(nz)
        
        zRange = range(self.nz-1, -1, -1) if 'topleft' in order.lower() else range(self.nz)
        yzMesh_np = np.array(np.meshgrid(range(self.ny), zRange)).transpose((1,2,0)).reshape((-1,2))
        yzMesh = torch.from_numpy(yzMesh_np).cuda()
        xyzMesh = torch.hstack( [torch.zeros((self.ny*self.nz,1)).cuda(), yzMesh])

        self.antType = antType
        self.centered = centered
        self.order = order

        if not centered:
            self.xyzs = antSpacingOverLambda*(xyzMesh)
        else:
            offset = [[0, (ny-1)/2, (nz-1)/2]]
            self.xyzs = antSpacingOverLambda*(xyzMesh-offset)
            
        if self.order[0] in 'zZ':
            # zTopLeft: 1  3    yTopLeft: 1  2  zBottomLeft: 2  4    yBottomLeft: 3  4
            #           2  4              3  4               1  3                 1  2
            self.xyzs = self.xyzs.reshape((nz,ny,3)).permute((1,0,2)).reshape(-1,3)

    @property
    def size(self):
        r"""The total number of antenna elements in the panel."""
        return torch.tensor(self.ny * self.nz)

    @property
    def shape(self):
        r"""The shape of the antenna panel as a tuple (nz, ny)."""
        return (self.nz,self.ny)

    def getResponse(self, phi, theta):        
        
        numPaths = phi.shape[-1]
        xyzPhases = torch.vstack([ torch.cos(phi)*torch.sin(theta),
                                torch.sin(phi)*torch.sin(theta),
                                torch.cos(theta)]).reshape(3,-1,numPaths)  # Shape: 3 x T x P

        #print(phi.shape,theta.shape)
        ar = torch.exp(2j * torch.pi * torch.sum(self.xyzs.T[:,None,:,None] * xyzPhases[:,:,None,:], axis=0)) # Shape: T, N, P
        
        # Shapes:  3,1,N,1 * 3,T,1,P => 3,T,N,P => Sum(axis 0) => T,N,P

        to_ret = ar/torch.sqrt(self.size)
        # Normalize the Array Response so that its norm at each time instance is equal to sqrt(P)
        return to_ret
         

    # *****************************************************************************************************************
    def getPattern(self, phi, theta):
        
        if self.antType == 'Isotropic': 
            return 1
        assert self.antType == 'halfwave-dipole'

        maxDipoleGain = 1.6409223769    # Half-wave dipole maximum directivity
        pattern = maxDipoleGain * np.square( np.cos((np.pi/2)*np.cos(theta))/np.sin(theta) )
        pattern[theta==0] = 0        # Handle the devision by zero cases (in our case theta should never be close to zero)
        return pattern



class Model:

    def __init__(self,numPaths,nt,nr):

        self.numPaths = numPaths
        numSubcarriers = 1
        self.u = 1
        self.txPower = 0.0
        
        self.kk = numSubcarriers                             # Total number of subcarriers
        subCarrierSpacing = (1<<self.u)*15000
        self.k = torch.from_numpy(np.arange(self.kk)*subCarrierSpacing).cuda()             # All subcarrier frequencies, Shape: K
        
        self.rxAnt = AntPanel(nt[0], nt[1], 'Isotropic') # Use linear antenna if only aoa and aod are used.
        self.txAnt = AntPanel(nr[0], nr[1], 'Isotropic')

        # Statistics for generated datasets.
        # Initialize Dictionary to save path information
        self.stats = dict()
        
        for n in range(numPaths):
            self.stats["path "+str(n)+" power"] = []
            self.stats["path "+str(n)+" aod"]   = []
            self.stats["path "+str(n)+" zod"]   = []
            self.stats["path "+str(n)+" aoa"]   = []
            self.stats["path "+str(n)+" zoa"]   = []
            self.stats["path "+str(n)+" phase"] = []
        
    def getChannel(self,gains, aoa,aod,zoa,zod):

        '''
        B = Batch size
        P = # of Paths
        
        All inputs are of shape B x P

        Output : Channel H (B x Nr x Nt x K)
        '''
        
        # Both powers and txPower are in db
        gains = gains + 1e-10
        gains = gains*self.rxAnt.getPattern(aoa,zoa) * self.txAnt.getPattern(aod,zod)  # Shape: B x P
        g = torch.sqrt(gains)      # Shape: B x P
        g = g*torch.sqrt((self.txAnt.size * self.rxAnt.size) / (self.numPaths * self.kk))

        txAr = self.txAnt.getResponse(aod,zod)          # Shape: B x Nt x P
        rxAr = self.rxAnt.getResponse(aoa,zoa)          # Shape: B x Nr x P
        
        # Shapes: B,P,1  *  B,P,1  *  1,1,K =>  B,P,K
        g = (g[:,:,None].clone())

        # Shapes: B,Nr,1,P,1 * B,1,Nt,P,1 * B,1,1,P,K => B,Nr,Nt,P,K => sum(axis=3) => B,Nr,Nt,K
        h = (rxAr[:,:,None,:,None] * txAr[:,None,:,:,None] * g[:,None,None,:,:]).sum(3) # Shape: B,Nr,Nt,K

        # Normalize only if needed!
        #h = h/torch.sqrt( torch.square(torch.abs(h)).mean((1,2)) )[:,None,None,:] # divide by the RMS of Nr x Nt matrixes
        
        # Convert to real domain and to 2-D Images.
        h = torch.stack((torch.real(h),torch.imag(h))).permute(1,0,2,3,4)[:,:,:,:,0]#.flatten(1,3)#[:,:,:,0]
    
        return h

    def getChannelfromArrayResponse(self,gains,txAr,rxAr):
        '''
        Get channel H given gains, Tx Response, Rx Response
        '''

        # Both powers and txPower are in db
        gains = gains + 1e-10
        g = torch.sqrt(gains)      # Shape: B x P
        g = g*torch.sqrt((self.txAnt.size * self.rxAnt.size) / (self.numPaths * self.kk))
        g = g[:,:,None]

        h = (rxAr[:,:,None,:,None] * txAr[:,None,:,:,None] * g[:,None,None,:,:]).sum(3) # Shape: B,Nr,Nt,K

        # Normalize only if needed
        #h = h/torch.sqrt( torch.square(torch.abs(h)).mean((1,2)) )[:,None,None,:] # divide by the RMS of Nr x Nt matrixes
        
        # Convert to real domain and to 2-D Images.
        h = torch.stack((torch.real(h),torch.imag(h))).permute(1,0,2,3,4)[:,:,:,:,0]#.flatten(1,3)#[:,:,:,0]

        return h

    def getChannelfromCombiArrayResponse(self,gains,combi_weights,combi_responses,use_gains=True,gain_proxy=1.0,num_paths=1):
        '''
        Get channel H given gain matrix, (Tx Response * Rx Response)
        '''
        
        if(use_gains):
            '''
            If gain values are given as inputs.
            '''
            
            # Both powers and txPower are in db
            gains = gains + 1e-10
            g = torch.sqrt(gains)      # Shape: B x P
            g = g*torch.sqrt((self.txAnt.size * self.rxAnt.size) / (num_paths * self.kk))
            g = g[:,:,None]
            print(g)

            h = (combiResponse * g[:,None,None,:,:]).sum(3) # Shape: T,Nr,Nt,K
        else:

            '''
            Using gains from the gain matrix
            '''
            
            # Turn Combi Responses into shape Batch x Num Ants x Resolution^2 to imitate antenna responses
            #combi_responses = torch.stack([combi_responses.squeeze().flatten(0,1).permute(1,2,0) for i in range(combi_weights.shape[0])]).unsqueeze(4)
            
            # Turn Combi Weights into Batch x Resolution^2 x 1
            combi_weights = 1e-5*combi_weights*torch.sqrt((self.txAnt.size * self.rxAnt.size) / (num_paths * self.kk)).type(torch.complex64)
            #combi_weights = combi_weights.squeeze().flatten(1)
            #combi_weights = combi_weights[:,:,None]

            #print(combi_weights.shape)
            #print(combi_responses.shape)
            
            # Multiply based on getChannel function method
            weighted_response = contract('xmnoyz,bmnocd->byz',combi_responses,combi_weights)#combi_responses*combi_weights[:,None,None,:,:]            
            h = weighted_response.unsqueeze(3) # Shape: B,Nr,Nt,K

        # Normalize only if needed!
        #h = h/torch.sqrt( torch.square(torch.abs(h)).mean((1,2)) )[:,None,None,:] # divide by the RMS of Nr x Nt matrixes
        
        # Convert to real domain and to 2-D Images.
        h = torch.stack((torch.real(h),torch.imag(h))).permute(1,0,2,3,4)[:,:,:,:,0]#.flatten(1,3)#[:,:,:,0]

        return h

    def getChannel_old(self,power, aod, zod, aoa, zoa, phase, tau):

        '''
        Legacy function from ChanSeq (Do not use)
        '''
        
        # Both powers and txPower are in db
        gains = gains*self.rxAnt.getPattern(aoa,zoa) * self.txAnt.getPattern(aod,zod)  # Shape: T x P
        
        # Apply normalization factors and doppler shift to the gains
        g = torch.sqrt(gains) * torch.exp(1j*phase)      # Shape: T x P
        g = g*torch.sqrt((self.txAnt.size * self.rxAnt.size) / (self.numPaths * self.kk))

        txAr = self.txAnt.getResponse(aod,zod).unsqueeze(1)              # Shape: T x Nt x P
        rxAr = self.rxAnt.getResponse(aoa,zoa).unsqueeze(1)              # Shape: T x Nr x P
        
        tau = tau[:,:,None].clone().permute(0,1,3,2)
        k = self.k[None,None,:].clone()
        k = torch.stack([k for i in range(power.shape[0])])
        g = (g[:,:,None].clone()).permute(0,1,3,2)
        
        g = g * torch.exp(-2j*torch.pi * tau * k)   # Shape: T,P,K
        
        # Shapes: T,Nr,1,P,1 * T,1,Nt,P,1 * T,1,1,P,K => T,Nr,Nt,P,K => sum(axis=3) => T,Nr,Nt,K
        h = (rxAr[:,:,:,None,:,None] * txAr[:,:,None,:,:,None] * g[:,None,None,:,:]).sum(4) # Shape: T,Nr,Nt,K
        h = h/torch.sqrt( torch.square(torch.abs(h)).mean((2,3)) )[:,None,None,:] # divide by the RMS of Nr x Nt matrixes

        # Convert to real domain and to 2-D Images.
        h = torch.stack((torch.real(h),torch.imag(h))).permute(1,0,2,3,4,5).flatten(1,4)
    
        return h

    def save_data(self,size=10000,set_name=None,dparams=None,save=True,batch=500):

        # Initialize structures to save channel information
        if(set_name==None or dparams==None):
            print('No Set Name/ Parameter Dict Given')
            return None

        channels = []
        full_dataset = dict()
        full_dataset['channel'] = []
        full_dataset['params'] = []

        self.numPaths = dparams['num_paths']

        if(dparams['deepmimo']):
            '''
            If using parameters generated by the DeepMIMO scenarios. (Stored by default in 'deepmimo_channels/')
            '''
            
            params = np.load('city_scenarios/param_datasets/'+dparams['datafolder']+'/params.npy')

            batch = len(params)

            # Read parameters from the file 
            path_power = torch.from_numpy(params[:,0,:]).float().cuda()#*(1e)
            path_aoa = torch.from_numpy(params[:,1,:]).float().cuda()
            path_aod = torch.from_numpy(params[:,2,:]).float().cuda()

            # Generate the rest of the parameters randomly. (By default set to static values.)
            path_zoa = torch.tensor(np.random.uniform(1.5708,1.5708,size=(batch,self.numPaths))).float().cuda()
            path_zod = torch.tensor(np.random.uniform(1.5708,1.5708,size=(batch,self.numPaths))).float().cuda()
            path_phase = torch.tensor(np.random.uniform(dparams['phase'][0],dparams['phase'][0]*dparams['unif_min_scale'],size=(batch,self.numPaths))).float().cuda()
            path_tau = torch.tensor(np.random.uniform(dparams['tau'][0],dparams['tau'][0]*dparams['unif_min_scale'],size=(batch,self.numPaths))).float().cuda()

            # Save the path statistics
            print(path_power.shape)
            for n in range(dparams['num_paths']):
                self.stats["path "+str(n)+" power"] += list(torch.clone(path_power[:,n]).flatten().cpu().detach().numpy())
                self.stats["path "+str(n)+" aoa"] += list(torch.clone(path_aoa[:,n]).flatten().cpu().detach().numpy())
                self.stats["path "+str(n)+" aod"] += list(torch.clone(path_aod[:,n]).flatten().cpu().detach().numpy())
                self.stats["path "+str(n)+" zoa"] += list(torch.clone(path_zoa[:,n]).flatten().cpu().detach().numpy())
                self.stats["path "+str(n)+" zod"] += list(torch.clone(path_zod[:,n]).flatten().cpu().detach().numpy())
                self.stats["path "+str(n)+" phase"] += list(torch.clone(path_phase[:,n]).flatten().cpu().detach().numpy())

            # Ensure that path gain > 0.0 in linear scale
            path_power = path_power*((path_power>0.0)*1)            
    
            # Generate channel
            model = Model(dparams['num_paths'],[dparams['num_ants'],1],[dparams['num_ants'],1])
            c = model.getChannel(path_power, path_aoa,path_aod,path_zoa,path_zod)
            
            channels.append(c)
            full_dataset['channel'].append(c.cpu().numpy())
            full_dataset['params'].append([path_power.cpu().numpy(),path_aoa.cpu().numpy(),path_aod.cpu().numpy(),path_zoa.cpu().numpy(),path_zod.cpu().numpy()])
            
        else:

            '''
            If generating channels randomly from a specified distribution of parameters.
            '''
            for i in range(size):
    
                power = []
                aoa = []
                aod = []
                zoa = []
                zod = []
                phase = []
                tau = []
    
                # Uniform Distribution
                for n in range(self.numPaths):
    
                    if(dparams['on_grid']):
                        '''
                        If generated channels should be from a discrete distribution.
                        '''
                        
                        path_aoa = torch.tensor(np.random.randint(dparams['aoa'][2*n],dparams['aoa'][2*n+1],size=(1,batch))).float().reshape(batch,1).cuda()
                        path_aod = torch.tensor(np.random.randint(dparams['aod'][2*n],dparams['aod'][2*n+1],size=(1,batch))).float().reshape(batch,1).cuda()
                        path_power = torch.tensor(np.random.uniform(dparams['pow'][n],dparams['pow'][n+1],size=(1,batch))).float().reshape(batch,1).cuda()
    
                        path_aoa = ((path_aoa+1)/dparams['resolution'])*2.0 - 1.0
                        path_aod = ((path_aod+1)/dparams['resolution'])*2.0 - 1.0
                        
                    else:
                        '''
                        If generated channels should be from a continuous distribution.
                        '''
                        
                        path_aoa = torch.tensor(np.random.uniform(dparams['aoa'][2*n]-dparams['aoa'][2*n+1],dparams['aoa'][2*n]+dparams['aoa'][2*n+1],size=(1,batch))).float().reshape(batch,1).cuda()
                        path_aod = torch.tensor(np.random.uniform(dparams['aod'][2*n]-dparams['aod'][2*n+1],dparams['aod'][2*n]+dparams['aod'][2*n+1],size=(1,batch))).float().reshape(batch,1).cuda()
                        path_power = torch.tensor(np.random.uniform(dparams['pow'][n],dparams['pow'][n+1],size=(1,batch))).float().reshape(batch,1).cuda()

                    # Generate the rest of the channels randomly (By default set to static values.)    
                    path_zoa = torch.tensor(np.random.uniform(dparams['zoa'][n],dparams['zoa'][n]*1.0,size=(1,batch))).float().reshape(batch,1).cuda()
                    path_zod = torch.tensor(np.random.uniform(dparams['zod'][n],dparams['zod'][n]*1.0,size=(1,batch))).float().reshape(batch,1).cuda()
                    path_phase = torch.tensor(np.random.uniform(dparams['phase'][n],dparams['phase'][n]*dparams['unif_min_scale'],size=(1,batch))).float().reshape(batch,1).cuda()
                    path_tau = torch.tensor(np.random.uniform(dparams['tau'][n],dparams['tau'][n]*dparams['unif_min_scale'],size=(1,batch))).float().reshape(batch,1).cuda()

                    # Save path parameters
                    power.append(path_power)
                    aoa.append(path_aoa)
                    aod.append(path_aod)
                    zoa.append(path_zoa)
                    zod.append(path_zod)
                    phase.append(path_phase)
                    tau.append(path_tau)
    
                    self.stats["path "+str(n)+" power"] += list(torch.clone(path_power).flatten().cpu().detach().numpy())
                    self.stats["path "+str(n)+" aoa"] += list(torch.clone(path_aoa).flatten().cpu().detach().numpy())
                    self.stats["path "+str(n)+" aod"] += list(torch.clone(path_aod).flatten().cpu().detach().numpy())
                    self.stats["path "+str(n)+" zoa"] += list(torch.clone(path_zoa).flatten().cpu().detach().numpy())
                    self.stats["path "+str(n)+" zod"] += list(torch.clone(path_zod).flatten().cpu().detach().numpy())
                    self.stats["path "+str(n)+" phase"] += list(torch.clone(path_phase).flatten().cpu().detach().numpy())

                power = torch.stack(power)[:,:,0].permute(1,0)
                aoa = torch.stack(aoa)[:,:,0].permute(1,0)
                aod = torch.stack(aod)[:,:,0].permute(1,0)
                zoa = torch.stack(zoa)[:,:,0].permute(1,0)
                zod = torch.stack(zod)[:,:,0].permute(1,0)
                phase = torch.stack(phase)[:,:,0].permute(1,0)
                tau = torch.stack(tau)[:,:,0].permute(1,0)

                power = power*((power>0.0)*1)            
                model = Model(dparams['num_paths'],[dparams['num_ants'],1],[dparams['num_ants'],1])
        
                # Simulate channel
                c = model.getChannel(power, aoa,aod,zoa,zod)
                channels.append(c)
                full_dataset['channel'].append(c.cpu().numpy())
                full_dataset['params'].append([power.cpu().numpy(),aoa.cpu().numpy(),aod.cpu().numpy(),zoa.cpu().numpy(),zod.cpu().numpy()])
            
        full_dataset['channel'] = np.vstack(full_dataset['channel'])
        full_dataset['params'] = np.hstack(full_dataset['params']).transpose(1,0,2)
        channels = torch.vstack(channels)

        # Get DFT representation of simulated channels
        dft_channels = torch.stack(self.get_dft(channels))
        dft_channels = torch.stack([torch.real(dft_channels),torch.imag(dft_channels)]).permute(1,0,2,3)
        full_dataset['dft_channel'] = dft_channels

        print('Channels Shape',full_dataset['channel'].shape)
        print('DFT Shape',full_dataset['dft_channel'].shape)
        print('Params Shape',full_dataset['params'].shape)
        
        # Split simulated data into training and validation sets.
        random_indices = [i for i in range(len(full_dataset['channel']))]
        random.shuffle(random_indices)
        training_indices = random_indices[0:int(0.8*len(random_indices))]
        validation_indices = random_indices[int(0.8*len(random_indices)):]

        full_dataset['training_channel'] = full_dataset['channel'][training_indices]
        full_dataset['training_dft'] = full_dataset['dft_channel'][training_indices]
        full_dataset['training_params'] = full_dataset['params'][training_indices]

        full_dataset['val_channel'] = full_dataset['channel'][validation_indices]
        full_dataset['val_dft'] = full_dataset['dft_channel'][validation_indices]
        full_dataset['val_params'] = full_dataset['params'][validation_indices]
        
        # Save Datasets.
        if(save):
            channels_np = channels.cpu().numpy()
            np.save('channel_data/'+set_name+'/'+set_name,channels_np)
            with open('channel_data/'+set_name+'/'+set_name+'_labelled.pkl', 'wb') as f:
                pickle.dump(full_dataset, f)
            print('Saved Dataset of size :',channels.shape)

        return full_dataset

    def get_dft(self,channels,set_name=None,dparams=None,save = False):
        # Input : Channels (Size x Real-Imag x Nt x Nr)
        
        # Convert to Complex array of shape Size x Nt x Nr
        channels = channels[:,0,:,:] + 1j*channels[:,1,:,:]

        # Create DFT matrices for Nt and Nr 
        fft_nr = torch.from_numpy(np.transpose(np.conjugate((1/(np.sqrt(channels.shape[1])))*np.array([[np.exp(2j*np.pi*i*p*1/channels.shape[1]) for i in range(channels.shape[1])] for p in range(channels.shape[1])]))).T).type(torch.complex64).cuda()
        fft_nt = torch.from_numpy((1/(np.sqrt(channels.shape[2])))*np.array([[np.exp(2j*np.pi*i*p*1/channels.shape[1]) for i in range(channels.shape[2])] for p in range(channels.shape[2])]).T).type(torch.complex64).cuda()
        
        # Apply DFT transform
        dft_ch = []
        channels = channels
        for idx in range(len(channels)): 
            dft_ch.append(fft_nr@channels[idx]@fft_nt)

        return dft_ch
