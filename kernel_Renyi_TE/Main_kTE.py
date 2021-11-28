# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 21:21:27 2021

@author: ide2704

Kernel-based phase transfer entropy

This script reproduces the results for the kernel-based Renyi transfer 
entropy method (alpha = 2) shown in Figure 2A of the paper "A data-driven 
measure of effective connectivity based on Renyi’s alpha-entropy". 

Ivan De La Pava Panche, Automatics Research Group
Universidad Tecnologica de Pereira, Pereira - Colombia
email: ide@utp.edu.co
"""

import os
import sys
import numpy as np
import scipy.io as sio
import TransferEntropy as TE

# Add current working directory to sys path 
sys.path.append(os.getcwd())

# =============================================================================
# %% Loading the simulated data 
# =============================================================================
data = sio.loadmat('VAR_data.mat')
VAR_data = data['VAR_data']
switch = data['myswitch'].flatten()

num_tests = VAR_data.shape[0]
num_trials = VAR_data.shape[1]
noise_levels = np.arange(0,1.1,0.1) # Noise levels

# =============================================================================
# %% Kernel transfer entropy estimation
# =============================================================================
kTE_lst = []
for i in range(num_tests):
    kTE_matrix = np.zeros((num_trials,len(noise_levels),2))
    for ii in range(num_trials): 
        print('Computing kernel TE, test {}/{}, trial {}/{}...'.format(i+1,num_tests,ii+1,num_trials))
        for iii in range(len(noise_levels)): 
            X = VAR_data[i,ii][:,:,iii]
        
            # kTE parameters 
            alpha = 2
            
            # Embedding time (autocorrelation decay time)
            maxlag = 20
            tau_x = TE.autocorr_decay_time(X[0,:],maxlag)
            tau_y = TE.autocorr_decay_time(X[1,:],maxlag) 
            tau_matrix = np.array([tau_x,tau_y])
            
            # Embedding dimension 
            dim = 3
            dim_matrix = np.array([dim,dim])
            
            # Interaction time (in samples)
            u = 1       
            
            # Estimating TE
            TE_aux = TE.kernelTransferEntropy_AllCh(X,dim_matrix,tau_matrix,u,alpha) 
            kTE_matrix[ii,iii,0] = TE_aux[0,1]
            kTE_matrix[ii,iii,1] = TE_aux[1,0]
            
    # Saving TE matrix for test i to list 
    kTE_lst.append(kTE_matrix)
    
# =============================================================================
# %% Accuracy (correct detection of the direction of interaction)
# =============================================================================

acc_matrix = np.zeros((num_tests,len(noise_levels)))
for j,kTE in enumerate(kTE_lst): 
    if switch[j]>0:
        acc_matrix[j,:] = np.mean(100*(kTE[:,:,0]-kTE[:,:,1]>0),axis=0)
    else:
        acc_matrix[j,:] = np.mean(100*(kTE[:,:,1]-kTE[:,:,0]>0),axis=0)
        
# =============================================================================
# %% Results plots
# =============================================================================
        
from matplotlib import pyplot as plt
plt.close('all')

plt.figure(figsize=[6.4, 5.2])
acc_avg = np.mean(acc_matrix,axis=0)
acc_std = np.std(acc_matrix,axis=0)
plt.fill_between(noise_levels,acc_avg-acc_std,acc_avg+acc_std,facecolor='b',alpha=0.2)
plt.plot(noise_levels,acc_avg,color='b',label=r'TE$_{\kappa\alpha}(\alpha = 2)$',linewidth=2)
plt.xlabel(r'$\gamma$',fontsize=16)
plt.ylabel('Accuracy (%)',fontsize=16)
plt.xlim([noise_levels[0],noise_levels[-1]])
plt.ylim([40,100])
plt.xticks(ticks=noise_levels,fontsize=13)
plt.yticks(fontsize=13)
plt.legend(fontsize=14,loc='best',frameon=False)
plt.savefig('kTE_VAR.png')  
