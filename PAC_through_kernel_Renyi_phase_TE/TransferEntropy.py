# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 21:06:13 2020

@author: ide2704

Kernel-based Renyi transfer entropy functions 

Ivan De La Pava Panche, Automatics Research Group
Universidad Tecnologica de Pereira, Pereira - Colombia
email: ide@utp.edu.co

"""
# Import the necessary libraries 
import numpy as np
import scipy.spatial as sp_spatial
from scipy.linalg import fractional_matrix_power

# =============================================================================
# Kernel-based Renyi transfer entropy 
# =============================================================================

def embeddingX(x,tau,dim,u):
    """
    Time-delay embbeding of the source time series x
    
    Parameters
    ----------
    x: ndarray of shape (samples,)
        Source time series 
    dim: int
        Embedding dimension 
    tau: int
        Embedding delay 
    u: int
        Interaction time

    Returns
    -------
    X_emb: ndarray of shape (samples-(tau*(dim-1))-u,dim)
        Time embedded source time series 
    """
    T = np.size(x)
    L = T -(dim-1)*tau
    firstP = T - L
    X_emb = np.zeros((L,dim))
    for i in range(L):
      for j in range(dim):
        X_emb[i,j] = x[i+firstP-(j*tau)]
    
    X_emb = X_emb[0:-u,:]
    return X_emb

def embeddingY(y,tau,dim,u):
    """
    Time-delay embbeding of the target time series y 
    
    Parameters
    ----------
    y: ndarray of shape (samples,)
        Target time series 
    dim: int
        Embedding dimension 
    tau: int
        Embedding delay 
    u: int
        Interaction time

    Returns
    -------
    Y_emb: ndarray of shape (samples-(tau*(dim-1))-u,dim)
        Time embedded target time series 
    y_t: ndarray of shape (samples-(tau*(dim-1))-u,1)
        Time shifted target time series 
    """
    T = np.size(y)
    L = T -(dim-1)*tau
    firstP = T - L
    Y_emb = np.zeros((L,dim))
    
    for i in range(L):
      for j in range(dim):
        Y_emb[i,j] = y[i+firstP-(j*tau)]
    
    y_t = y[firstP+u::] 
    y_t = y_t.reshape(y_t.shape[0],1)
    Y_emb = Y_emb[u-1:-1,:]
    return Y_emb,y_t

def GaussianKernel(X,sig_scale=1.0):
    """
    Compute Gaussian Kernel matrix    
   
    Parameters
    ----------
    X: ndarray of shape (samples,features)
        Input data 
    sig_scale: float
        Parameter to scale the kernel's bandwidth  
        
    Returns
    -------
    K: ndarray of shape (samples,samples)
        Gaussian kernel matrix
    """
    utri_ind =  np.triu_indices(X.shape[0], 1)
    dist = sp_spatial.distance.cdist(X,X,'euclidean')
    sigma = sig_scale*np.median(dist[utri_ind])
    K = np.exp(-1*(dist**2)/(2*sigma**2))
    return K

def kernelRenyiEntropy(K_lst,alpha):
    """
    Compute Renyi's entropy from kernel matrices 
    
    Parameters
    ----------
    K_lst: list
        List holding kernel matrices [ndarrays of shape (channels,channels)]
    alpha: int or float
        Order of Renyi's entropy
        
    Returns
    -------
    h: float
        Kernel-based Renyi's transfer entropy, TE(x->y)
    """
    if len(K_lst) == 1:
      K = K_lst[0]
    elif len(K_lst) == 2:
      K = K_lst[0]*K_lst[1]
    else:
      K = K_lst[0]*K_lst[1]*K_lst[2]     
    K = K/np.trace(K) 
    if alpha%1==0:
        h = np.real((1/(1-alpha))*np.log2(np.trace(np.linalg.matrix_power(K,alpha))))
    else:
        h = np.real((1/(1-alpha))*np.log2(np.trace(fractional_matrix_power(K,alpha))))
    return h

def kernelTransferEntropy(x,y,dim,tau,u,alpha,sig_scale=1.0): 
    """
    Compute kernel-based Renyi's transfer entropy from channel x to channel y
    
    Parameters
    ----------
    x: ndarray of shape (samples,)
        Source time series 
    y: ndarray of shape (samples,)
        Target time series 
    dim: int
        Embedding dimension 
    tau: int
        Embedding delay 
    u: int
        Interaction time
    alpha: int or float
        Order of Renyi's entropy
    sig_scale: float
        Parameter to scale the kernel's bandwidth  

    Returns
    -------
    TE: float
        Kernel-based Renyi's transfer entropy, TE(x->y)
    """
    dim = int(dim)
    tau = int(tau)
    u = int(u)
    
    X_emb = embeddingX(x,tau,dim,u)
    Y_emb, y_t = embeddingY(y,tau,dim,u)
    
    K_X_emb = GaussianKernel(X_emb,sig_scale)
    K_Y_emb = GaussianKernel(Y_emb,sig_scale)
    K_y_t = GaussianKernel(y_t,sig_scale)    
    
    h1 = kernelRenyiEntropy([K_X_emb,K_Y_emb],alpha)
    h2 = kernelRenyiEntropy([K_X_emb,K_Y_emb,K_y_t],alpha)
    h3 = kernelRenyiEntropy([K_Y_emb,K_y_t],alpha)
    h4 = kernelRenyiEntropy([K_Y_emb],alpha)
    
    TE = h1 - h2 + h3 - h4
    return TE

def kernelTransferEntropy_AllCh(X,Dim,Tau,u,alpha,sig_scale=1.0): 
    """
    Compute kernel-based Renyi's transfer entropy among all channels in X 
    
    Parameters
    ----------
    X: ndarray of shape (channels,samples)
        Input time series (number of channels x number of samples)
    Dim: ndarray of shape (channels,)
        Embedding dimension for each channel
    Tau: ndarray of shape (channels,)
        Embedding delay for each channel
    u: int
        Interaction time
    alpha: int or float
        Order of Renyi's entropy
    sig_scale: float
        Parameter to scale the kernel's bandwidth  

    Returns
    -------
    TE: ndarray of shape (channels,channels) 
        Kernel-based Renyi's transfer entropy for all channel pairs in X
    """
    num_ch = X.shape[0]
    TE = np.zeros([num_ch,num_ch])
    for j in range(num_ch):
        # Target channel  
        y = X[j,:]  
        # Embedding parameters 
        tau = int(Tau[j])
        dim = int(Dim[j])  
        u = int(u)
        # Time embeddings for y 
        Y_emb, y_t = embeddingY(y,tau,dim,u)
        # Kernels for y's time embeddings 
        K_Y_emb = GaussianKernel(Y_emb,sig_scale)
        K_y_t = GaussianKernel(y_t,sig_scale)   
        # Entropies 
        h3 = kernelRenyiEntropy([K_Y_emb,K_y_t],alpha)
        h4 = kernelRenyiEntropy([K_Y_emb],alpha)
        for i in range(num_ch):
            if i!=j:
                # Source channel 
                x = X[i,:]
                # Time embedding for x 
                X_emb = embeddingX(x,tau,dim,u)
                # Kernels for x's time embedding
                K_X_emb = GaussianKernel(X_emb,sig_scale)
                # Entropies 
                h1 = kernelRenyiEntropy([K_X_emb,K_Y_emb],alpha)
                h2 = kernelRenyiEntropy([K_X_emb,K_Y_emb,K_y_t],alpha)
                # Transfer entropy
                TE[i,j] =  h1 - h2 + h3 - h4
    return TE

def kernelTransferEntropy_AllCh_freq(X,Dim,Tau,u_matrix,alpha,freq,time,component='phase',sig_scale=1.0): 
    """
    Compute kernel-based Renyi's phase transfer entropy among all channels in X 
    
    Parameters
    ----------
    X: ndarray of shape (channels,samples)
        Input time series (number of channels x number of samples)
    Dim: ndarray of shape (channels,)
        Embedding dimension for each channel
    Tau: ndarray of shape (channels,)
        Embedding delay for each channel
    u_matrix: ndarray of shape (channels,channels)
        Matrix holding the interaction times for each channel pair 
    alpha: int or float
        Order of Renyi's entropy
    freq: float 
        Frequency of interest for phase extraction, in Hz
    time: ndarray of shape (samples,)
        Time vector (must be sampled at the sampling frequency of X)
    component: {'filt','amp','phase'}
       Component of interest from the wavelet decomposition at the frequency 
       in freq (filt: filtered data, amp: amplitude envelope, phase: phase)
    sig_scale: float
        Parameter to scale the kernel's bandwidth  

    Returns
    -------
    TE_ph: ndarray of shape (channels,channels) 
        Kernel-based Renyi's phase transfer entropy for all channel pairs in X
        at frequency freq
    """
    num_ch = X.shape[0]
    TE_ph = np.zeros([num_ch,num_ch])
    
    # Wavelet decomposition 
    X_ph = Wavelet_Trial_Dec(X,time,freq,component)
    
    for j in range(num_ch):
        # Embedding parameters 
        tau = int(Tau[j])
        dim = int(Dim[j])  
        for i in range(num_ch):
            if i!=j:
                # Delay time 
                u = int(u_matrix[i,j])
            
                # Target channel  
                y = X_ph[j,:]  
                # Source channel 
                x = X_ph[i,:]
                
                # Time embeddings for y 
                Y_emb, y_t = embeddingY(y,tau,dim,u)
                # Kernels for y's time embeddings 
                K_Y_emb = GaussianKernel(Y_emb,sig_scale)
                K_y_t = GaussianKernel(y_t,sig_scale)   
                # Entropies 
                h3 = kernelRenyiEntropy([K_Y_emb,K_y_t],alpha)
                h4 = kernelRenyiEntropy([K_Y_emb],alpha)
        
                # Time embedding for x 
                X_emb = embeddingX(x,tau,dim,u)
                # Kernels for x's time embedding
                K_X_emb = GaussianKernel(X_emb,sig_scale)
                # Entropies 
                h1 = kernelRenyiEntropy([K_X_emb,K_Y_emb],alpha)
                h2 = kernelRenyiEntropy([K_X_emb,K_Y_emb,K_y_t],alpha)
               
                # Transfer entropy
                TE_ph[i,j] =  h1 - h2 + h3 - h4
    return TE_ph
                
def kernelTransferEntropy_PAC(X,Dim,Tau,U,alpha,freq_ph,freq_amp,time,sig_scale=1.0): 
    """
    Compute directed phase-amplitude interactions through kernel-based Renyi's phase transfer
    entropy between a pair channels 
    
    Parameters
    ----------
    X: ndarray of shape (channels,samples)
        Input time series (number of channels x number of samples), the number of channels must be 2 
    Dim: ndarray of shape (channels,)
        Embedding dimension for each channel
    Tau: ndarray of shape (channels,)
        Embedding delay for each channel
    U: ndarray of shape (channels,)
        Interaction times for each channel pair and direction of interaction
    alpha: int or float
        Order of Renyi's entropy
    freq_ph: ndarray of shape (frequencies_ph,)
        Frequencies of interest for phase extraction, in Hz
    freq_amp: ndarray of shape (frequencies_amp,)
        Frequencies of interest for amplitude extraction, in Hz
    time: ndarray of shape (samples,)
        Time vector (must be sampled at the sampling frequency of X)
    sig_scale: float
        Parameter to scale the kernel's bandwidth  

    Returns
    -------
    TE_pac: ndarray of shape (frequencies_ph,frequencies_amp,2) 
        Directed PAC estimated through kernel-based Renyi's phase transfer
        entropy (TE_pac[:,:,0] holds interactions from ch1->ch2, 
        TE_pac[:,:,1] holds interactions from ch2->ch1)
    """
    num_freq_ph = np.size(freq_ph)
    num_freq_amp = np.size(freq_amp)
    TE_pac = np.zeros([num_freq_ph,num_freq_amp,2])
    ind_matrix = np.array([[0,1],[1,0]])
        
    for k in range(2):
        tau = Tau[k]
        dim = Dim[k]
        u = U[k]
        indices = ind_matrix[k,:]
        src = X[indices[0],:]
        src = src.reshape((1,len(src)))
        trg = X[indices[1],:]
        trg = trg.reshape((1,len(trg)))
    
        # Wavelet decomposition 
        src_ph = Wavelet_Trial_Dec(src,time,freq_ph,component='phase')[0,:,:]
        src_ph = src_ph.T
        trg_amp = Wavelet_Trial_Dec(trg,time,freq_amp,component='amp')[0,:,:]
        trg_amp = trg_amp.T
        
        trg_amp_ph_lst = []
        for kk in range(num_freq_amp):
            trg_amp_aux = trg_amp[kk,:].reshape((1,len(trg_amp[kk,:])))
            if (trg.shape[1] % 2) == 0:
                time = time.flatten()
                trg_amp_ph_lst.append(np.transpose(Wavelet_Trial_Dec(trg_amp_aux,time[:-1],freq_ph,component='phase')[0,:,:]))
            else:
                trg_amp_ph_lst.append(np.transpose(Wavelet_Trial_Dec(trg_amp_aux,time,freq_ph,component='phase')[0,:,:]))
        
        for j in range(num_freq_amp):
            # Embedding parameters 
            tau = int(tau)
            dim = int(dim)  
            for i in range(num_freq_ph):
                # Delay time 
                u = int(u)
            
                # Target channel  
                y = trg_amp_ph_lst[j][i,:]
                # Source channel 
                x = src_ph[i,:]
                
                # Time embeddings for y 
                Y_emb, y_t = embeddingY(y,tau,dim,u)
                # Kernels for y's time embeddings 
                K_Y_emb = GaussianKernel(Y_emb,sig_scale)
                K_y_t = GaussianKernel(y_t,sig_scale)   
                # Entropies 
                h3 = kernelRenyiEntropy([K_Y_emb,K_y_t],alpha)
                h4 = kernelRenyiEntropy([K_Y_emb],alpha)
        
                # Time embedding for x 
                X_emb = embeddingX(x,tau,dim,u)
                # Kernels for x's time embedding
                K_X_emb = GaussianKernel(X_emb,sig_scale)
                # Entropies 
                h1 = kernelRenyiEntropy([K_X_emb,K_Y_emb],alpha)
                h2 = kernelRenyiEntropy([K_X_emb,K_Y_emb,K_y_t],alpha)
               
                # Transfer entropy
                TE_pac[i,j,k] =  h1 - h2 + h3 - h4
    return TE_pac

# =============================================================================
# Embedding functions 
# =============================================================================

def autocorrelation(x):
    """
    Autocorrelation of x
    
    Parameters
    ----------
    x: ndarray of shape (samples,)
        Input time series 

    Returns
    -------
    act: ndarray of shape (samples,) 
        Autocorrelation 
    """
    xp = (x - np.mean(x))/np.std(x)
    result = np.correlate(xp, xp, mode='full')
    auto_corr = result[int(result.size/2):]/len(xp)
    return auto_corr

def autocorr_decay_time(x,maxlag):
    """ 
    Autocorrelation decay time (embedding delay)
    
    Parameters
    ----------
    x: ndarray of shape (samples,)
        Input time series 
    maxlag: int
        Maximum embedding delay

    Returns
    -------
    act: int 
        Embedding delay  
    """
    autocorr = autocorrelation(x)
    thresh = np.exp(-1)
    aux = autocorr[0:maxlag];
    aux_lag = np.arange(0,maxlag)
    if len(aux_lag[aux<thresh]) == 0:
        act = maxlag
    else:
        act = np.min(aux_lag[aux<thresh])
    return act

def cao_criterion(x,d_max,tau):
    """ 
    Cao's criterion (embedding dimension)
    
    Parameters
    ----------
    x: ndarray of shape (samples,)
        Input time series 
    d_max: int
        Maximum embedding dimension 
    tau: int
        Embedding delay

    Returns
    -------
    dim: int 
        Embedding dimension 
    """
    tau = int(tau)
    N = len(x)
    d_max = int(d_max)+1 
    x_emb_lst = []
    
    for d in range(d_max):
        # Time embedding 
        T = np.size(x)
        L = T-(d*tau)
        if L>0:
            FirstP = T-L
            x_emb = np.zeros((L,d+1))
            for ii in range(0,L):
                for jj in range(0,d+1): 
                    x_emb[ii,jj] = x[ii+FirstP-(jj*tau)]
            x_emb_lst.append(x_emb)
    
    d_aux = len(x_emb_lst)
    E = np.zeros(d_aux-1)
    for d in range(d_aux-1):
        emb_len = N-((d+1)*tau)
        a = np.zeros(emb_len)
        for i in range(emb_len): 
            var_den = x_emb_lst[d][i,:]-x_emb_lst[d][0:emb_len,:]
            inf_norm_den = np.linalg.norm(var_den,np.inf,axis=1)
            inf_norm_den[inf_norm_den==0] = np.inf
            den = np.min(inf_norm_den)
            ind = np.argmin(inf_norm_den)
            num = np.linalg.norm(x_emb_lst[d+1][i,:]-x_emb_lst[d+1][ind,:],np.inf)
            a[i] = num/den
        E[d] = np.sum(a)/emb_len
    
    E1 = np.roll(E,-1)  # circular shift
    E1 = E1[:-1]/E[:-1]
    
    dim_aux = np.zeros([1,len(E1)-1])
    
    for j in range(1,len(E1)-1):
        dim_aux[0,j] = E1[j-1]+E1[j+1]-2*E1[j]
    dim_aux[dim_aux==0] = np.inf
    dim = np.argmin(dim_aux)+1

    return dim

# =============================================================================
# Wavelet transform
# =============================================================================

def Morlet_Wavelet(data,time,freq):
    """
    Morlet wavelet decomposition 
    
    Parameters
    ----------
    data: ndarray of shape (samples,)
        Input signal 
    time: ndarray of shape (samples,)
        Time vector (must be sampled at the sampling frequency of data, 
        best practice is to have time=0 at the center of the wavelet)
    freq: ndarray of shape (frequencies,)
        Frequencies to evaluate in Hz
        
    Returns
    -------
    dataW: dict of keys {'amp','filt','phase','f'}
        Dictionary containing the Morlet wavelet decomposition of data
        'amp': ndarray of shape (frequencies,num_samples) holding the amplitude envelopes at each freq
        'filt': ndarray of shape (frequencies,num_samples) holding the filtered signals at each freq
        'phase': ndarray of shape (frequencies,num_samples) holding the phase time series at each freq
        'f': ndarray of shape (frequencies,) holding the evaluated frequencies in Hz
        (If samples is odd, num_samples = samples, otherwise num_samples = samples-1)
    """
    # =============================================================================
    # Create the Morlet wavelets
    # =============================================================================
    num_freq = len(freq); 
    cmw = np.zeros([data.shape[0],num_freq],dtype = 'complex_')
    
    # Number of cycles in the wavelets 
    range_cycles = [3,10]
    max_freq = 60 
    freq_vec = np.arange(1,max_freq+1)
    nCycles_aux = np.logspace(np.log10(range_cycles[0]),np.log10(range_cycles[-1]),len(freq_vec))
    nCycles = np.array([nCycles_aux[np.argmin(np.abs(freq_vec - freq[i]))] 
                        for i in range(num_freq)])
    
    for ii in range(num_freq): 
        # create complex sine wave
        sine_wave = np.exp(1j*2*np.pi*freq[ii]*time)
        
        # create Gaussian window
        s = nCycles[ii]/(2*np.pi*freq[ii]) #this is the standard deviation of the gaussian
        gaus_win  = np.exp((-time**2)/(2*s**2))
        
        # now create Morlet wavelet
        cmw[:,ii] = sine_wave*gaus_win
        
    # =============================================================================
    # Convolution 
    # =============================================================================
    
    # Define convolution parameters 
    nData = len(data)
    nKern = cmw.shape[0]
    nConv = nData + nKern - 1
    half_wav = int(np.floor(cmw.shape[0]/2)+1)
    
    # FFTs
    
    # FFT of wavelet, and amplitude-normalize in the frequency domain
    cmwX = np.fft.fft(cmw,nConv,axis=0)
    cmwX = cmwX/np.max(cmwX,axis=0)
        
    # FFT of data
    dataX = np.fft.fft(data,nConv)
    dataX = np.repeat(dataX.reshape([-1,1]),num_freq,axis=1)
    
    # Convolution...
    data_wav = np.fft.ifft(dataX*cmwX,axis=0)
    
    # Cut 1/2 of the length of the wavelet from the beginning and from the end
    data_wav = data_wav[half_wav-2:-half_wav,:]
    
    # Extract filtered data, amplitude and phase 
    data_wav = data_wav.T
    dataW = {}
    dataW['filt'] = np.real(data_wav)
    dataW['amp'] = np.abs(data_wav)
    dataW['phase'] = np.angle(data_wav)
    dataW['f'] = freq 
    
    return dataW

def Wavelet_Trial_Dec(data,time,freq,component ='phase'):
    """
    Morlet wavelet decomposition for multiple channels 
    
    Parameters
    ----------
    data: ndarray of shape (channels,samples)
        Input signals (number of channels x number of samples)
    time: ndarray of shape (samples,)
        Time vector (must be sampled at the sampling frequency of data)
    freq: ndarray of shape (frequencies,)
        Frequencies to evaluate in Hz
    component: {'filt','amp','phase'}
       Component of interest from the wavelet decomposition at each frequency 
       in freq (filt: filtered data, amp: amplitude envelope, phase: phase)

    Returns
    -------
    wav_dec: ndarray of shape (channels,num_samples,frequencies) 
        Array containing the wavelet decomposition of data at the target
        frequencies (If samples is odd, num_samples = samples, otherwise 
        num_samples = samples-1)

    """
    if np.size(freq) == 1:
        freq = [freq]

    # Time centering 
    t = (time.flatten())/1000 # ms to s
    t = t - t[0]
    t = t-(t[-1]/2) # best practice is to have time=0 at the center of the wavelet

    if (data.shape[1] % 2) == 0:
        wav_dec = np.zeros([data.shape[0],data.shape[1]-1,len(freq)])
    else:
        wav_dec = np.zeros([data.shape[0],data.shape[1],len(freq)])

    for ch in range(data.shape[0]):
       
        # Data detrending
        ch_data = data[ch,:]
        ch_data = ch_data - np.mean(ch_data)
        
        # Data decomposition 
        dataW = Morlet_Wavelet(ch_data,t,freq)
        wav_dec[ch,:,:] = dataW[component].T
    
    return wav_dec
            
# =============================================================================
# Permutation test 
# =============================================================================

def permutation_test(Te,Te_sh,alpha):
    '''
    Permutation test using trial randomized surrogates
    
    Parameters
    ----------
    Te: ndarray of shape (n,2)
        nx2 connectivity matrix, where n is the number of trials. The rows
        of Te hold effective connectivity values between 2 channels x and y.
        For instance, the first row of TE holds elements of the form
        [TE(x1->y1),TE(y1->x1)], where x1 and y1 stand for channels x and y
        of trial 1.
    Te_sh: ndarray of shape (n,2)
        nx2 shuffled connectivity matrix, where n is the number of trials. The rows
        of Te_sh hold effective connectivity values between 2 channels x and y.
        For instance, the first row of TE_sh holds elements of the form
        [TE(x1->y2),TE(y1->x2)], where x1 and y1 stand for channels x and y
        of trial 1, and x2 and y2 stand for channels x and y of trial 2.  
    alpha: float 
        alpha level of the permutation test (usually 0.05 or 0.01).
     
    Returns
    -------
    Tepermvalues: ndarray of shape (2,)
        Value of the permutation test for TE(x->y) and TE(y->x)
    signigicance: ndarray of shape (2,)
        Statistical significance of TE(x->y) and TE(y->x) (based on the value of the
        permutation test and the stablished alpha level)  
    dTepermvalues: float  
        Value of the permutation test for net TE (TE(x->y)-TE(y->x))
    dsignigicance: float 
        Statistical significance for net TE (TE(x->y)-TE(y->x))   
    '''
    mean_TE = np.mean(Te,axis=0)
    mean_TE_sh = np.mean(Te_sh,axis=0)
    Testatistic = np.abs(mean_TE-mean_TE_sh)

    dTE = Te[:,0]-Te[:,1]
    mean_dTE = np.mean(dTE)
    dTE_sh = Te_sh[:,0] - Te_sh[:,1]
    mean_dTE_sh = np.mean(dTE_sh)
    dTestatistic =  np.abs(mean_dTE-mean_dTE_sh)
    
    # Permutation #
    #nrcmc = 2
    n = Te.shape[0]
    m = Te_sh.shape[0]
    numperm = 10000
    dist_1 = np.zeros((numperm,Te.shape[1]))
    dist_2 = np.zeros((numperm,Te_sh.shape[1]))
    
    ddist_1 = np.zeros(numperm)
    ddist_2 = np.zeros(numperm)
    
    for l in range(numperm):
        data_pool = np.concatenate((Te,Te_sh),axis=0)
        # permuting indexes from 1 to n+m
        ind = np.argsort(np.random.rand(1,n+m))
        ind_1 = ind[0,:n]
        data_1 = data_pool[ind_1,:]
        ind_2 = ind[0,n:n+m]
        data_2 = data_pool[ind_2,:]
        dist_1[l,:] = np.mean(data_1,axis=0)
        dist_2[l,:] = np.mean(data_2,axis=0)
        
        data_pool_d = np.concatenate((dTE,dTE_sh))
        ddist_1[l] = np.mean(data_pool_d[ind_1])
        ddist_2[l] = np.mean(data_pool_d[ind_2])
        
    Tepermdist = np.abs(dist_1-dist_2) 
    dTepermdist = np.abs(ddist_1-ddist_2)
    
    Tepermvalues = np.zeros(2)
    significance = np.zeros(2)
    for i in range(2):
        Tepermvalues[i] = len(np.where(Tepermdist[:,i]>Testatistic[i])[0])/numperm
        significance[i] = Tepermvalues[i]<=alpha
    
    dTepermvalues = len(np.where(dTepermdist>dTestatistic)[0])/numperm
    dsignificance = float(dTepermvalues<=alpha)
        
    return Tepermvalues,significance,dTepermvalues,dsignificance