

"""
calculate the geometric medians, cosine distance, NBR and NBRdistance for the given data over the long term period
return a xarray with packed geometric median in [bands,y,x], cosdist, NBRdist, NBR in [time,y,x], together with the threshold used for filtering cosdist and NBR dist in the change detection in [y,x]
"""

import numpy as np
import xarray as xr
from burnmappingtoolbox import geometric_median, cosdistance, nbr_eucdistance

def nbr(data):
    """
    calculate the normalised burn ration for the given 6 bands landsat data (cloudy free,outputs from loaddea module)
    """
    NBR = (data.nir-data.swir2)/(data.nir+data.swir2)
    return NBR

def geomedian_and_dists(data):   
    #calculate the Geomatric Median for the whole period and the cosine distance relative to the geomatric median
    MaxInter = 60
    tol      = 1.e-7
    Nbands = 6
    GeoMed= np.empty((Nbands,len(data.y),len(data.x))) # geometric medians
    GeoMed.fill(np.nan)
    cosdist = np.empty((len(data.time),len(data.y),len(data.x))) # cosine distance
    cosdist.fill(np.nan)
    NBR = nbr(data) # get NBR time series
    NBRdist = np.empty((len(data.time),len(data.y),len(data.x))) # NBR euclidean distance
    NBRdist.fill(np.nan)
    Sign = np.zeros((len(data.time),len(data.y),len(data.x))) # NBR change direction
    NBRmed = np.empty((len(data.y),len(data.x))) # NBR geometric median
    NBRmed.fill(np.nan)

    for y in range(0,len(data.y)):
        #construct a p x N matric X including the 
        X = np.empty((Nbands,len(data.time),len(data.x)))
        X[0,:,:] = data.blue[:,y,:] 
        X[1,:,:] = data.green[:,y,:]
        X[2,:,:] = data.red[:,y,:]
        X[3,:,:] = data.nir[:,y,:]
        X[4,:,:] = data.swir1[:,y,:]
        X[5,:,:] = data.swir2[:,y,:]
        X[X<=0] = np.nan
        #calculation of geometric median for each row
        GeoMed[:,y,:] = np.transpose(np.vstack([geometric_median(X[:,:,x],tol,MaxInter) for x in range(0,len(data.x))]))

        #calculation of cosine distance for each row
        cosdist[:,y,:] = np.transpose(np.vstack([cosdistance(GeoMed[:,y,x],X[:,:,x]) for x in range(0,len(data.x))]))

        #calculation of NBR distance for each row
        NBRmed[y,:] = (GeoMed[3,y,:]-GeoMed[5,y,:])/(GeoMed[3,y,:]+GeoMed[5,y,:])    
        tmp = np.vstack([nbr_eucdistance(NBRmed[y,x],NBR.data[:,y,x]) for x in range(0,len(data.x))])
        NBRdist[:,y,:]=np.transpose((tmp[0:len(data.x)*2:2]))
        Sign[:,y,:]=np.transpose((tmp[1:len(data.x)*2:2]))
    # calculate the outliers for change detection as (75th percentile + 1.5*IQR)
    NBRoutlier=np.nanpercentile(NBRdist,75,axis=0)+1.5*(np.nanpercentile(NBRdist,75,axis=0)-np.nanpercentile(NBRdist,25,axis=0))
    CDistoutlier=np.nanpercentile(cosdist,75,axis=0)+1.5*(np.nanpercentile(cosdist,75,axis=0)-np.nanpercentile(cosdist,25,axis=0))
    ds = xr.Dataset({'GeoMed':(('bands','y','x'),GeoMed[:]),'NBR':(('time','y','x'),NBR[:]),'CosDist':(('time','y','x'),cosdist[:]),
                'NBRDist':(('time','y','x'),NBRdist[:]),'NegtiveChange':(('time','y','x'),Sign[:]),
                'NBRoutlier':(('y','x'),NBRoutlier[:]),'CDistoutlier':(('y','x'),CDistoutlier)},
               coords={'time':data.time[:],'y':data.y[:],'x':data.x[:],'bands':np.linspace(0,5,6)},attrs={'crs':'EPSG:3577'})
    return ds


