#!/usr/bin/env python3
"""
@author: meysamhashemi  INS Marseille

"""
import os
import sys
import numpy as np
import scipy as scp

from scipy import signal
from scipy import stats as spstats
from scipy.signal import hilbert
from scipy.stats import moment, mode, skew, kurtosis


from scipy.optimize import fsolve
from scipy.optimize import root

from numpy import linalg as LA
from sklearn.decomposition import PCA


from FCD import extract_FCD
######################################################
def calculate_summary_statistics(x, nn, features):
    """Calculate summary statistics

    Parameters
    ----------
    x : output of the simulator

    Returns
    -------
    np.array, summary statistics
    """
    
        

    X=x.reshape(nn, int(x.shape[0]/nn))

    n_summary = 16*nn+(nn*nn)+300*300

    fs = 10e3 

    wwidth=30
    maxNwindows=200
    olap=0.94
        
    sum_stats_vec = np.concatenate((np.mean(X, axis=1), 
                                    np.median(X, axis=1),
                                    np.std(X, axis=1),
                                    skew(X, axis=1), 
                                    kurtosis(X, axis=1),
                                    ))
   


    for item in features:

            if item is 'higher_moments':

                        sum_stats_vec = np.concatenate((sum_stats_vec,
                                                moment(X, moment=2, axis=1),
                                                moment(X, moment=3, axis=1),
                                                moment(X, moment=4, axis=1),
                                                moment(X, moment=5, axis=1),
                                                moment(X, moment=6, axis=1),
                                                moment(X, moment=7, axis=1),
                                                moment(X, moment=8, axis=1),
                                                moment(X, moment=9, axis=1), 
                                                moment(X, moment=10, axis=1),        
                                                                   ))
                        
            if item is 'spectral_power':

                        f, Pxx_den =  signal.periodogram(X, fs)

                        sum_stats_vec = np.concatenate((sum_stats_vec,
                                        np.max(Pxx_den, axis=1), 
                                        np.mean(Pxx_den, axis=1),
                                        np.median(Pxx_den, axis=1),
                                        np.std(Pxx_den, axis=1),
                                        skew(Pxx_den, axis=1), 
                                        kurtosis(Pxx_den, axis=1), 
                                        np.diag(np.dot(Pxx_den, Pxx_den.transpose())),
                                                       ))

       
            if item is 'envelope':

                        analytic_signal = hilbert(X)
                        amplitude_envelope = np.abs(analytic_signal)
                        instantaneous_phase = np.unwrap(np.angle(analytic_signal))

                        sum_stats_vec = np.concatenate((sum_stats_vec,
                                        np.mean(amplitude_envelope, axis=1),
                                        np.std(amplitude_envelope, axis=1),
                                        np.mean(instantaneous_phase, axis=1),
                                        np.std(instantaneous_phase, axis=1),
                                                       ))
                            


            if item is 'FC_corr':

                        FCcorr=np.corrcoef(X)
                        off_diag_sum_FC = np.sum(FCcorr) - np.trace(FCcorr)
                        FC_TRIU = np.triu(FCcorr, k=10)
                        eigen_vals_FC, _ = LA.eig(FCcorr)
                        pca = PCA(n_components=3)
                        PCA_FC = pca.fit_transform(FCcorr)

                        Upper_FC = []
                        Lower_FC = []
                        for i in range(0,len(FCcorr)):
                            Upper_FC.extend(FCcorr[i][i+1:])
                            Lower_FC.extend(FCcorr[i][0:i])

                        sum_stats_vec = np.concatenate((sum_stats_vec,
                                        np.array([np.sum(eigen_vals_FC.reshape(-1))]),                                         
                                        np.array([np.max(eigen_vals_FC.reshape(-1))]),  
                                        np.array([np.min(eigen_vals_FC.reshape(-1))]),  
                                        np.array([np.mean(eigen_vals_FC.reshape(-1))]),  
                                        np.array([np.std(eigen_vals_FC.reshape(-1))]),  
                                        np.array([skew(eigen_vals_FC.reshape(-1))]),
                                        np.array([kurtosis(eigen_vals_FC.reshape(-1))]),  
                                                        
                                        np.array([np.sum(PCA_FC.reshape(-1))]),                                         
                                        np.array([np.max(PCA_FC.reshape(-1))]),  
                                        np.array([np.min(PCA_FC.reshape(-1))]),  
                                        np.array([np.mean(PCA_FC.reshape(-1))]),  
                                        np.array([np.std(PCA_FC.reshape(-1))]),  
                                        np.array([skew(PCA_FC.reshape(-1))]),
                                        np.array([kurtosis(PCA_FC.reshape(-1))]),  

                                        np.array([np.sum(Upper_FC)]),                                         
                                        np.array([np.max(Upper_FC)]),  
                                        np.array([np.min(Upper_FC)]),  
                                        np.array([np.mean(Upper_FC)]),  
                                        np.array([np.std(Upper_FC)]),  
                                        np.array([skew(Upper_FC)]),
                                        np.array([kurtosis(Upper_FC)]), 
                                                        
                                        np.array([np.sum(Lower_FC)]),                                         
                                        np.array([np.max(Lower_FC)]),  
                                        np.array([np.min(Lower_FC)]),  
                                        np.array([np.mean(Lower_FC)]),  
                                        np.array([np.std(Lower_FC)]),  
                                        np.array([skew(Lower_FC)]),
                                        np.array([kurtosis(Lower_FC)]), 

                                        np.array([np.sum(FC_TRIU.reshape(-1))]),
                                        np.array([np.max(FC_TRIU.reshape(-1))]),
                                        np.array([np.min(FC_TRIU.reshape(-1))]),
                                        np.array([np.mean(FC_TRIU.reshape(-1))]),
                                        np.array([np.std(FC_TRIU.reshape(-1))]),
                                        np.array([skew(FC_TRIU.reshape(-1))]),
                                        np.array([kurtosis(FC_TRIU.reshape(-1))]),
                                                        
                                        np.array([off_diag_sum_FC]),

                                                       ))

                        
            if item is 'FCD_corr':

                        FCDcorr,Pcorr,shift=extract_FCD(X,wwidth,maxNwindows,olap,mode='corr')
                        off_diag_sum_FCD = np.sum(FCDcorr) - np.trace(FCDcorr)
                        FCD_TRIU = np.triu(FCDcorr, k=10)
                        
                        eigen_vals_FCD, _ = LA.eig(FCDcorr)
                        pca = PCA(n_components=3)
                        PCA_FCD = pca.fit_transform(FCDcorr)
                        
                        Upper_FCD = []
                        Lower_FCD = []
                        for i in range(0,len(FCDcorr)):
                            Upper_FCD.extend(FCDcorr[i][i+1:])
                            Lower_FCD.extend(FCDcorr[i][0:i])

                    
                        sum_stats_vec = np.concatenate((sum_stats_vec,
                                        np.array([np.sum(eigen_vals_FCD.reshape(-1))]),                                         
                                        np.array([np.max(eigen_vals_FCD.reshape(-1))]),  
                                        np.array([np.min(eigen_vals_FCD.reshape(-1))]),  
                                        np.array([np.mean(eigen_vals_FCD.reshape(-1))]),  
                                        np.array([np.std(eigen_vals_FCD.reshape(-1))]),  
                                        np.array([skew(eigen_vals_FCD.reshape(-1))]),
                                        np.array([kurtosis(eigen_vals_FCD.reshape(-1))]),  
                                                        
                                        np.array([np.sum(PCA_FCD.reshape(-1))]),                                         
                                        np.array([np.max(PCA_FCD.reshape(-1))]),  
                                        np.array([np.min(PCA_FCD.reshape(-1))]),  
                                        np.array([np.mean(PCA_FCD.reshape(-1))]),  
                                        np.array([np.std(PCA_FCD.reshape(-1))]),  
                                        np.array([skew(PCA_FCD.reshape(-1))]),
                                        np.array([kurtosis(PCA_FCD.reshape(-1))]),  

                                        np.array([np.sum(Upper_FCD)]),                                         
                                        np.array([np.max(Upper_FCD)]),  
                                        np.array([np.min(Upper_FCD)]),  
                                        np.array([np.mean(Upper_FCD)]),  
                                        np.array([np.std(Upper_FCD)]),  
                                        np.array([skew(Upper_FCD)]),
                                        np.array([kurtosis(Upper_FCD)]), 
                                                        
                                        np.array([np.sum(Lower_FCD)]),                                         
                                        np.array([np.max(Lower_FCD)]),  
                                        np.array([np.min(Lower_FCD)]),  
                                        np.array([np.mean(Lower_FCD)]),  
                                        np.array([np.std(Lower_FCD)]),  
                                        np.array([skew(Lower_FCD)]),
                                        np.array([kurtosis(Lower_FCD)]), 
                                                        
                                        np.array([np.sum(FCD_TRIU.reshape(-1))]),
                                        np.array([np.max(FCD_TRIU.reshape(-1))]),
                                        np.array([np.min(FCD_TRIU.reshape(-1))]),
                                        np.array([np.mean(FCD_TRIU.reshape(-1))]),
                                        np.array([np.std(FCD_TRIU.reshape(-1))]),
                                        np.array([skew(FCD_TRIU.reshape(-1))]),
                                        np.array([kurtosis(FCD_TRIU.reshape(-1))]),
                                                        
                                        np.array([off_diag_sum_FCD]),                                                        
                                                        
                                                       ))
            
    sum_stats_vec = sum_stats_vec[0:n_summary]        


    return sum_stats_vec
