



import numpy as np
import math
"""
The geomedian module include three functions:
geomatric_median, medoid and spectral_angle
"""
def geometric_median(X,tol,MaxInter):
    """
    This is a script to calculate the Geomatric Median 
    The required variables are:
    "X" is a p x N matric, wehre p = number of bands and N = the number of dates in the period of interest
    "MaxNIter" is the maximum number of iteration
    "tol" is tolerance
    The procedure stop when EITHER error tolerance in solution 'tol' or the maximum number of iteration 'MaxNIter' is reached. 
    Returns a p-dimensional vector   'geoMedian' 
    """
    NDATES = len(X[0])
    l = 0
    y0 = np.nanmean(X,axis=1)
    if len(y0[np.isnan(y0)])==NDATES:
        geoMedian = y0
    else:
        eps = 10**2
        while ( np.sqrt(np.sum(eps**2))> tol and l< MaxInter):

            EucDist = np.transpose(np.transpose(X) - y0)
            EucNorm = np.sqrt(np.sum(EucDist**2,axis=1))
            NotNaN = np.where(~np.isnan(EucNorm))[0]
            y1=np.sum(X[:,NotNaN]/EucNorm[NotNaN],axis=1)/(np.sum(1/EucNorm[NotNaN]))
            if len(y1[~np.isnan(y1)])!=NDATES:
                eps = 0
            else:
                eps = y1 - y0
                y0 = y1
                l = l+1
        geoMedian = y0
        
    return geoMedian


def spectral_angle(ref,obs):
    """
    'ref' is the reference spectrum, p-dimentional
    'obs' is an arbitary observed spectrum, o-dimensional
    returns the Spetral Angle (in degrees); return NAN if obs have any NAN.
    """
    numer = np.sum(ref*obs)
    denom = np.sqrt(sum(ref**2)*sum(obs**2))
    if ~np.isnan(numer) and ~np.isnan(denom):
        alpha = np.arccos(numer/denom)*180./math.pi
    else:
        alpha = np.nan
    return alpha


def medoid(X):
    """
    "X" is a p x N matric, wehre p = number of bands and N = the number of dates in the period of interest
    Returns a p-dimensional vector Medoid
    """
    NDATES = len(X[0])
    TrackSum = np.empty((NDATES));
    TrackSum.fill(np.nan)
    y0 = np.nanmean(X,axis=1)
    EucDist = np.transpose(np.transpose(X)-y0)
    EucNorm = np.sqrt(np.sum(EucDist**2,axis=0))
    NotNaN = np.where(~np.isnan(EucNorm))[0]
    if len(NotNaN)>0:
        for counter,ll in enumerate(NotNaN):
            y0 = X[:,ll]
            EucDist = np.transpose(np.transpose(X) - y0)
            EucNorm = np.sqrt(np.sum(EucDist**2,axis=0))
            TrackSum[ll] = np.nansum(EucNorm)
        indmin = np.where(TrackSum==np.nanmin(TrackSum))[0]
        Med = X[:,indmin[0]]
       
    else:
        Med = y0
        
    return Med