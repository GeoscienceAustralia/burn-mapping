


"""
This module include two functions for distance between observation and reference: Cosdistance and Eucdistance
Required inputs:
ref: reference e.g. geomatric median [Nbands]
obs: observation e.g. monthly geomatric median or reflectance [Nbands,ndays]
return distance at each time step [ndays]
"""

import numpy as np


def Cosdistance(ref,obs):
    cosdist = np.empty((obs.shape[1],))
    cosdist.fill(np.nan)
    index = np.where(~np.isnan(obs[0,:]))[0]
    tmp = [1 - np.sum(ref*obs[:,t])/(np.sqrt(np.sum(ref**2))*np.sqrt(np.sum(obs[:,t]**2))) for t in index]
    cosdist[index] = np.transpose(tmp)
    return cosdist


def Eucdistance(ref,obs):
    EucDist = np.transpose(np.transpose(obs) - ref)
    EucNorm = np.sqrt(np.sum(EucDist**2,axis=0))
    
    return EucNorm






