"""
This module includes all functions used in the burn-severity mapping method. Functions are:

geometric_median(X,tol,MaxIter): to calculate the geometric median of band reflectances over a given period

spectral_angle(ref,obs): to calculate the spectral angle to the reference

medoid(X): to calculate the medoid over the given period for the given point

cosdistance(ref,obs): to calculate the cosine distance from the observation to the reference for the given point

nbr_eucdistance(ref,obs): to calculate the euclidean distance between NBR and NBRmed
(NBR calculated from geometric median)
"""

import numpy as np
import math


def geometric_median(x, tol, max_iter):
    """
    Calculates the geometric median of band reflectances
    The procedure stops when either the error tolerance 'tol' or the maximum number of iterations 'MaxIter' is reached. 

    Args:
        x: (p x N) matrix, where p = number of bands and N = number of dates during the period of interest
        max_iter: maximum number of iterations
        tol: tolerance criterion to stop iteration   
    
    Returns:
        geo_median: p-dimensional vector with geometric median reflectances
    """
    i = 0
    y0 = np.nanmean(x, axis=1)
    if len(y0[np.isnan(y0)]) > 0:
        return y0

    eps = 10 ** 2
    while np.sqrt(np.sum(eps ** 2)) > tol and i < max_iter:

        euc_dist = np.transpose(np.transpose(x) - y0)
        euc_norm = np.sqrt(np.sum(euc_dist ** 2, axis=0))
        not_nan = np.where(~np.isnan(euc_norm))[0]
        y1 = np.sum(x[:, not_nan] / euc_norm[not_nan], axis=1) / (np.sum(1 / euc_norm[not_nan]))
        if len(y1[np.isnan(y1)]) > 0:
            eps = 0
        else:
            eps = y1 - y0
            y0 = y1
            i += 1
    geo_median = y0

    return geo_median


def spectral_angle(ref, obs):
    """
    Calculates the spectral angle between two reflectance signatures
    
    Args:
        ref: reference spectrum (p-dimensional)
        obs: arbitrary observed spectrum (o-dimensional)
    
    Returns:
        alpha: spectral angle (in degrees)
    
    Note: 
        returns NAN if there are any NAN in ref or obs.
    """
    numer = np.sum(ref * obs)
    denom = np.sqrt(sum(ref ** 2) * sum(obs ** 2))
    if ~np.isnan(numer) and ~np.isnan(denom):
        alpha = np.arccos(numer / denom) * 180. / math.pi
    else:
        alpha = np.nan
    return alpha


def medoid(x):
    """
    Returns medoid of x
    
    Args:
        x: p x N matrix with data, where p = number of bands and N = the number of dates in the period of interest
    
    Returns:
        Med: medoid (p-dimensional vector)
    """
    n_dates = len(x[0])
    track_sum = np.empty(n_dates)
    track_sum.fill(np.nan)
    y0 = np.nanmean(x, axis=1)
    euc_dist = np.transpose(np.transpose(x) - y0)
    euc_norm = np.sqrt(np.sum(euc_dist ** 2, axis=0))
    not_nan = np.where(~np.isnan(euc_norm))[0]
    if len(not_nan) > 0:
        for counter, ll in enumerate(not_nan):
            y0 = x[:, ll]
            euc_dist = np.transpose(np.transpose(x) - y0)
            euc_norm = np.sqrt(np.sum(euc_dist ** 2, axis=0))
            track_sum[ll] = np.nansum(euc_norm)
        indmin = np.where(track_sum == np.nanmin(track_sum))[0]
        med = x[:, indmin[0]]

    else:
        med = y0

    return med


def cos_distance(ref, obs):
    """
    Returns the cosine distance between observation and reference

    The calculation is point based, easily adaptable to any dimension. 

    Args:
        ref: reference (2-D array with multiple days) e.g., geomatrix median [Nbands]
        obs: observation (with multiple bands, e.g. 6) e.g.,  monthly geomatrix median or reflectance [Nbands,ndays]
    
    Returns:
        cosdist: the cosine distance at each time step in [ndays]
    """
    cosdist = np.empty((obs.shape[1],))
    cosdist.fill(np.nan)
    index = np.where(~np.isnan(obs[0, :]))[0]
    tmp = [1 - np.sum(ref * obs[:, t]) / (np.sqrt(np.sum(ref ** 2)) * np.sqrt(np.sum(obs[:, t] ** 2))) for t in index]
    cosdist[index] = np.transpose(tmp)
    return cosdist


def nbr_eucdistance(ref, obs):
    """
    Returns the euclidean distance between the NBR at each time step with the NBR calculated from the geometric medians
    and also the direction of change to the NBR from the geometric medians.
    
    Args:
        ref: NBR calculated from geometric median, one value
        obs: NBR time series, 1-D time series array with ndays 
    
    Returns:
        NBRdist: the euclidean distance 
        Sign: change direction (1: decrease; 0: increase) at each time step in [ndays]
    """
    nbr_dist = np.empty((obs.shape[0],))
    sign = np.zeros((obs.shape[0],))
    nbr_dist.fill(np.nan)
    index = np.where(~np.isnan(obs[:]))[0]
    euc_dist = (obs[index] - ref)
    euc_norm = np.sqrt((euc_dist ** 2))
    nbr_dist[index] = euc_norm
    sign[index[euc_dist < 0]] = 1
    return nbr_dist, sign


def severity(c_dist, c_dist_outlier, time, nbr, nbr_dist, nbr_outlier, sign, method=3):
    """
    Returns the severity,duration and start date of the change.

    FIXME: bad code smell here with the `method` parameter. suggest refactoring into seperate functions

    Args:
        c_dist: cosine distance in 1D list with ndays
        c_dist_outlier: outlier for cosine distance, 1 for each point
        time: dates
        nbr: NBR time series in 1D list with ndays
        nbr_dist: Euclidean distance from NBR to NBRmed in a 1D list with ndays
        nbr_outlier: outlier for NBRdist
        sign: change direction for NBR, if Sign==1, NBR decrease from the median
        method: 1,2,3 to choose
            1: only use cosine distance as an indicator for change
            2: use cosine distance together with NBR<0
            3: use both cosine distance, NBR euclidean distance, and NBR change direction for change detection
        
    Returns:
        sevindex: severity
        startdate: first date change was detected 
        duration: duration between the first and last date the change exceeded the outlier threshold
    """
    sevindex = 0
    startdate = 0
    duration = 0
    notnanind = np.where(~np.isnan(c_dist))[0]  # remove the nan values for each pixel
    cosdist = c_dist[notnanind]

    if method == 1:  # only include change where cosine distance above the line
        outlierind = np.where((cosdist > c_dist_outlier))[0]
    elif method == 2:  # cosdist above the line and NBR<0
        outlierind = np.where((cosdist > c_dist_outlier) & (nbr[notnanind] < 0))[0]
    elif method == 3:  # both cosdist and NBR dist above the line and it is negative change
        outlierind = np.where((cosdist > c_dist_outlier) &
                              (nbr_dist[notnanind] > nbr_outlier) &
                              (sign[notnanind] == 1))[0]
    else:
        raise ValueError

    time = time[notnanind]
    outlierdates = time[outlierind]
    n_out = len(outlierind)
    area_above_d0 = 0
    if n_out >= 2:
        tt = []
        for ii in range(0, n_out):
            if outlierind[ii] + 1 < len(time):
                u = np.where(time[outlierind[ii] + 1] == outlierdates)[0]  # next day have to be outlier to be included
                # print(u)

                if len(u) > 0:
                    t1_t0 = (time[outlierind[ii] + 1] - time[outlierind[ii]]) / np.timedelta64(1, 's') / (60 * 60 * 24)
                    y1_y0 = (cosdist[outlierind[ii] + 1] + cosdist[outlierind[ii]]) - 2 * c_dist_outlier
                    area_above_d0 = area_above_d0 + 0.5 * y1_y0 * t1_t0  # calculate the area under the curve
                    duration = duration + t1_t0
                    tt.append(ii)  # record the index where it is detected as a change

        if len(tt) > 0:
            startdate = time[outlierind[tt[0]]]  # record the date of the first change
            sevindex = area_above_d0

    return sevindex, startdate, duration
