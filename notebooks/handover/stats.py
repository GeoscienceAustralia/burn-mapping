"""
This module includes all functions used in the burn-severity mapping method. Functions are:
geometric_median(X,tol,MaxIter): to calculate the geometric median of band reflectances over a given period
cos_distance(ref,obs): to calculate the cosine distance from the observation to the reference for the given point
nbr_eucdistance(ref,obs): to calculate the euclidean distance between NBR and NBRmed
(NBR calculated from geometric median)
severity(c_dist, c_dist_outlier, time, nbr, nbr_dist, nbr_outlier, sign, method=3): to calculate the severity index, start date and duration
"""
import numpy as np
import math
def geometric_median(x, epsilon, max_iter):
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
    y0 = np.mean(x, axis=1)
    if len(y0) == 0:
        return np.zeros((6))

    
    for _ in range(max_iter):
        euc_dist = np.transpose(np.transpose(x) - y0)
        euc_norm = np.sqrt(np.sum(euc_dist ** 2, axis=0))
        #not_nan = np.where(~np.isnan(euc_norm))[0]
        y1 = np.sum(x / euc_norm, axis=1) / (np.sum(1 / euc_norm))
        if len(y1[np.isnan(y1)]) > 0 or np.sqrt(np.sum((y1 - y0) ** 2)) < epsilon:
            return y1
      
        y0 = y1
        
    return int(y0)

def cos_distance(ref, obs):
    """
    Returns the cosine distance between observation and reference
    The calculation is point based, easily adaptable to any dimension. 
    Args:
        ref: reference (2-D array with multiple days) e.g., geomatric median [Nbands]
        obs: observation (with multiple bands, e.g. 6) e.g.,  monthly geomatric median or reflectance [Nbands,ndays]
    
    Returns:
        cosdist: the cosine distance at each time step in [ndays]
    """
    cosdist = np.empty((obs.shape[1]))
    cosdist.fill(np.nan)           
    index = np.where(~np.isnan(obs[0, :]))[0]
   
    from scipy import spatial
    for t in index: 
        #print(ref,obs[:,t])
        cosdist[t] = spatial.distance.cosine(ref, obs[:,t])    
        #print(cosdist[t])
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
    direction = np.zeros((obs.shape[0],),dtype='uint16')
    nbr_dist.fill(np.nan)
    index = np.where(np.isnan(obs))[0]
    euc_dist = (obs[index] - ref)
    euc_norm = np.sqrt((euc_dist ** 2))
    nbr_dist[index] = euc_norm
    direction[index[euc_dist< -0.05]] = 1
    
    return nbr_dist, direction

def severity(time, data, method=3):
    """
    Returns the severity,duration and start date of the change.
    FIXME: bad code smell here with the `method` parameter. suggest refactoring into seperate functions
    Args:
        c_dist: cosine distance in 1D list with ndays
        c_dist_outlier: outlier for cosine distance, 1 for each point
        time: dates of observations
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
    notnanind = np.where(~np.isnan(data.CosDist))[0]  # remove the nan values for each pixel
   

    if method == 1:  # only include change where cosine distance above the line
        try:
            data.CosDist
            data.CDistoutlier
        except NameError:
            print('Cosine distance is required')
            
        
        outlierind = np.where((data.CosDist[notnanind] > data.CDistoutlier))[0]
        cosdist = data.CosDist[notnanind]
    elif method == 2:  # cosdist above the line and NBR<0
        try:
            data.CosDist
            data.NBR            
            data.CDistoutlier
        except NameError:
            print('Cosine distance and NBR are required')
            
        
        outlierind = np.where((data.CosDist[notnanind] > data.CDistoutlier) & (data.NBR[notnanind] < 0))[0]
        cosdist = data.CosDist[notnanind]
    elif method == 3:  # both cosdist and NBR dist above the line and it is negative change
        try:
            data.CosDist
            data.NBRDist
            data.ChangeDir
            data.CDistoutlier
            data.NBRoutlier
        except NameError:
            print('Cosine distance are NBR distance are required')
        
        
        outlierind = np.where((data.CosDist[notnanind] > data.CDistoutlier) &
                              (data.NBRDist[notnanind] > data.NBRoutlier) &
                              (data.ChangeDir[notnanind] == 1))[0]

        cosdist = data.CosDist.data[notnanind]
    else:
        raise ValueError

    time = time.data[notnanind]
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
                    y1_y0 = (cosdist[outlierind[ii] + 1] + cosdist[outlierind[ii]]) - 2 * data.CDistoutlier.data
                    area_above_d0 = area_above_d0 + 0.5 * y1_y0 * t1_t0  # calculate the area under the curve
                    duration = duration + t1_t0
                    tt.append(ii)  # record the index where it is detected as a change

        if len(tt) > 0:
            startdate = time[outlierind[tt[0]]]  # record the date of the first change
            sevindex = area_above_d0

    return sevindex, startdate, duration
