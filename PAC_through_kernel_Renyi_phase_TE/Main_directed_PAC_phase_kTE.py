# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 21:57:40 2021

@author: ide2704

Directed PAC through kernel based Phase Transfer Entropy

This script reproduces the results shown in Figure 4 of the paper "Estimating 
directed phase-amplitude interactions from EEG data through kernel-based 
phase transfer entropy". 

Ivan De La Pava Panche, Automatics Research Group
Universidad Tecnologica de Pereira, Pereira - Colombia
email: ide@utp.edu.co
"""

import os
import sys
import numpy as np
import scipy.io as sio
from joblib import Parallel,delayed
import TransferEntropy as TE

# Add current working directory to sys path 
sys.path.append(os.getcwd())

# =============================================================================
# %% Loading the simulated data 
# =============================================================================
SNR = 3
m = 0.25
data_path = 'Directed_PAC_SimData_STR{}_m{}.mat'.format(str(SNR),str(m).replace(".",""))
CFC_data = sio.loadmat(data_path)
f_sample = CFC_data['fs'][0].astype('float')  # Sampling frequency 
t = CFC_data['t'].flatten()
data = CFC_data['data']

num_trials = data.shape[0]
freq_ph = np.arange(3,48,3) # frequency of wavelet (phase), in Hz 
freq_amp = np.arange(3,48,3) # frequency of wavelet (amplitude), in Hz 
num_freq_ph = len(freq_ph)
num_freq_amp = len(freq_amp)

# Creating shuffled data for statistical testing 
ind_shuffle = np.roll(np.arange(0,num_trials),-1)
data_sh = data[ind_shuffle,:,:]

u_vec = np.arange(4,44,4)
tau_matrix = np.zeros((num_trials,2))
dim_matrix = np.zeros((num_trials,2))
u_matrix = np.zeros((num_trials,2))
kTE_u_matrix = np.zeros((num_trials,len(u_vec),2))
kTE_pac_lst = []
kTE_pac_sh_lst = []

# =============================================================================
# %% Estimating directed phase-amplitude interactions through kernel phase TE
# =============================================================================
for k in range(num_trials):
    print('SNR: {}, m: {}, Trial {}/{}'.format(SNR,m,k+1,num_trials))
    
    # Data downsampling (fs: 1000 Hz -> 250 Hz)
    n = 4
    X = data[k,:,:]
    X = X[:,::n]
    X_sh = data_sh[k,:,:]
    X_sh = X_sh[:,::n]
    t_vec = 1000*t
    t_vec = t_vec[::n]
    fsample = f_sample/n
    
    # kTE parameters 
    
    alpha = 2
    
    print('Estimating embedding parameters...')          
    # Embedding time (autocorrelation decay time)
    maxlag = 20
    tau_matrix[k,0] = TE.autocorr_decay_time(X[0,:],maxlag)
    tau_matrix[k,1] = TE.autocorr_decay_time(X[1,:],maxlag) 
    
    # Embedding dimension (obtained using the cao criterion) 
    d_max = 10
    dim_matrix[k,0] = TE.cao_criterion(X[0,:],d_max,tau_matrix[k,0])
    dim_matrix[k,1] = TE.cao_criterion(X[1,:],d_max,tau_matrix[k,1]) 

    # Estimating interaction time 
    print('Computing kernel TE (to estimate u)...')
    TE_aux = Parallel(n_jobs=-1,verbose=0)(delayed(TE.kernelTransferEntropy_AllCh) 
                                    (X,dim_matrix[k,:],tau_matrix[k,:],u,alpha) for u in u_vec)
    
    TE_aux = np.transpose(np.array(TE_aux), (1,2,0))
    kTE_u_matrix[k,:,0] = TE_aux[0,1,:]
    kTE_u_matrix[k,:,1] = TE_aux[1,0,:]
    u_matrix[k,:] = u_vec[np.argmax(kTE_u_matrix[k,:,:],axis=0)]
    
    print('Estimating directed PAC using kernel TE...')   
    kTE_pac_aux = Parallel(n_jobs=-1,verbose=0)(delayed(TE.kernelTransferEntropy_PAC) 
                                    (X,dim_matrix[k,:],tau_matrix[k,:],u_matrix[k,:],alpha,freq_ph,f_amp,t_vec) for f_amp in freq_amp)
    kTE_pac = np.array(kTE_pac_aux)
    kTE_pac = np.transpose(kTE_pac[:,:,0,:],(1,0,2))
    kTE_pac_lst.append(kTE_pac)
    
    print('Estimating directed PAC (shuffled data) using kernel TE...')
    kTE_pac_sh_aux = Parallel(n_jobs=-1,verbose=0)(delayed(TE.kernelTransferEntropy_PAC) 
                                    (np.vstack((X[0,:],X_sh[1,:])),dim_matrix[k,:],tau_matrix[k,:],u_matrix[k,:],alpha,freq_ph,f_amp,t_vec) for f_amp in freq_amp)
    kTE_pac_sh = np.array(kTE_pac_sh_aux)
    kTE_pac_sh = np.transpose(kTE_pac_sh[:,:,0,:],(1,0,2))
    kTE_pac_sh_lst.append(kTE_pac_sh)

# =============================================================================
# %% Running permutation tests
# =============================================================================
print('\nRunning permutation tests ...')

alpha_level = 0.01  #Significance level for the test 
nr2cmc = num_freq_ph*num_freq_amp   #number of tests (For Bonferroni correction)

kTEpermval = np.zeros((num_freq_ph,num_freq_amp,2))
kTEsignif = np.zeros((num_freq_ph,num_freq_amp,2)) 

for i in range(num_freq_ph):
    print('Phase frequency: {} Hz'.format(freq_ph[i]))
    for j in range(num_freq_amp):
        kTE_aux = np.array(kTE_pac_lst)[:,i,j,:]
        kTE_sh_aux = np.array(kTE_pac_sh_lst)[:,i,j,:]
        # Permutation test
        kTEpermval[i,j,:],kTEsignif[i,j,:],dkTEpermval,dkTEsignif = TE.permutation_test(kTE_aux,kTE_sh_aux,alpha_level/nr2cmc)
        
# =============================================================================
# %% Plotting the obtained results          
# =============================================================================
from matplotlib import pyplot as plt 
plt.close('all')

kTE_pac_Xph_Yamp = np.mean(np.array(kTE_pac_lst)[:,:,:,0],axis=0)
kTE_pac_Yph_Xamp = np.mean(np.array(kTE_pac_lst)[:,:,:,1],axis=0)
max_val = np.max([kTE_pac_Xph_Yamp.max(),kTE_pac_Yph_Xamp.max()])
min_val = np.min([kTE_pac_Xph_Yamp.min(),kTE_pac_Yph_Xamp.min()])

plt.figure()
plt.imshow(kTE_pac_Xph_Yamp, vmin=min_val, vmax=max_val)
plt.xticks(np.arange(len(freq_amp)),freq_amp,rotation=45,fontsize=12)
plt.yticks(np.arange(len(freq_ph)),freq_ph,rotation=0,fontsize=12)
plt.ylabel(r'$f_{l}$ (Hz)',fontsize=16)
plt.xlabel(r'$f_{h}$ (Hz)',fontsize=16)
plt.tight_layout()
plt.savefig('Figure_4A.png') 

plt.figure()
plt.imshow(kTE_pac_Yph_Xamp, vmin=min_val, vmax=max_val)
plt.xticks(np.arange(len(freq_amp)),freq_amp,rotation=45,fontsize=12)
plt.yticks(np.arange(len(freq_ph)),freq_ph,rotation=0,fontsize=12)
plt.ylabel(r'$f_{l}$ (Hz)',fontsize=16)
plt.xlabel(r'$f_{h}$ (Hz)',fontsize=16)
plt.tight_layout()
plt.savefig('Figure_4B.png') 

plt.figure()
plt.imshow(1-kTEsignif[:,:,0], vmin=0, vmax=1,cmap='binary')
plt.xticks(np.arange(len(freq_amp)),freq_amp,rotation=45,fontsize=12)
plt.yticks(np.arange(len(freq_ph)),freq_ph,rotation=0,fontsize=12)
plt.ylabel(r'$f_{l}$ (Hz)',fontsize=16)
plt.xlabel(r'$f_{h}$ (Hz)',fontsize=16)
plt.tight_layout()
plt.savefig('Figure_4C.png') 

plt.figure()
plt.imshow(1-kTEsignif[:,:,1], vmin=0, vmax=1,cmap='binary')
plt.xticks(np.arange(len(freq_amp)),freq_amp,rotation=45,fontsize=12)
plt.yticks(np.arange(len(freq_ph)),freq_ph,rotation=0,fontsize=12)
plt.ylabel(r'$f_{l}$ (Hz)',fontsize=16)
plt.xlabel(r'$f_{h}$ (Hz)',fontsize=16)
plt.tight_layout()
plt.savefig('Figure_4D.png') 