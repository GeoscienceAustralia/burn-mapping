
import sys
import time
import pickle
import numpy as np
import xarray as xr
from stats import geometric_median, cos_distance, severity, nbr_eucdistance


def geomedian(data, names=['red','green','blue','nir','swir1','swir2'], epsilon=1, max_iter=40):
    """
    Calculates the geometric median of band reflectances
    The procedure stops when either the error tolerance 'tol' or the maximum number of iterations 'MaxIter' is reached. 
    Args:
        data: (p x Y x X) matrix, where p = number of bands and Y x X is the size of the data 
        name (optional): name of each band
        max_iter (optional): maximum number of iterations
        tol (optional): tolerance criterion to stop iteration   
    
    Returns:
        ds: xarray including geomedian, cosdist and outlier value for cosdist
    """
    
    X = np.empty((len(names),len(data.time),len(data.x)*len(data.y)))
    for i, name in enumerate(names):
        X[i,:,:] = data[name].data.reshape(len(data.time), -1)
    #X[X<=0] = np.nan
    

    gmed = np.zeros((len(names),len(data.y)*len(data.x))) # geometric medians
    #gmed.fill(np.nan)
    
    CDistoutlier = np.empty((len(data.y)*len(data.x)))
    CDistoutlier.fill(np.nan)
    NBRoutlier = np.empty((len(data.y)*len(data.x)))
    NBRoutlier.fill(np.nan)
    for i in range(0,len(data.y)*len(data.x)):        
        ind=[j for j in range(0,len(data.time)) if X[1,j,i]>0]
        
        if len(ind)>0:
            #print('-aa')
            x = X[:,ind,i]
            gmed[:,i] = geometric_median(x, epsilon, max_iter)
            cosdist = np.zeros((len(ind)))
            #print(gmed[:,i].dtype)
            cosdist = cos_distance(gmed[:,i], X[:,ind,i])
            NBR = np.empty((len(ind)))
            NBR = (x[3,:]-x[5,:])/(x[3,:]-x[5,:])
            NBRmed = (gmed[3,i]-gmed[5,i])/(gmed[3,i]+gmed[5,i])
            NBRdist = np.empty((len(ind)))
            direction = np.empty((len(ind)))
            NBRdist,direction = nbr_eucdistance(NBRmed,NBR)
            CDistoutlier[i] = np.nanpercentile(cosdist,75)+1.5*(np.nanpercentile(cosdist,75)-np.nanpercentile(cosdist,25))
            NBRoutlier[i] = np.nanpercentile(NBRdist,75)+1.5*(np.nanpercentile(NBRdist,75)-np.nanpercentile(NBRdist,25))
            
    ds = xr.Dataset(coords={'y':data.y[:],'x':data.x[:],'bands':names}, attrs={'crs':'EPSG:3577'})
    ds['geomedian'] = (('bands','y','x'),gmed[:].reshape((len(names),len(data.y),len(data.x))).astype('float32'))
    #ds['cosdist'] = (('time','y','x'),cosdist[:].reshape((len(data.time),len(data.y),len(data.x))))
    ds['CDistoutlier'] = (('y','x'),CDistoutlier.reshape((len(data.y),len(data.x))).astype('float32'))
    ds['NBRoutlier'] = (('y','x'),NBRoutlier.reshape((len(data.y),len(data.x))).astype('float32'))
    return ds

def cosdist(data, geomedian):
    names=['red','green','blue','nir','swir1','swir2']
    X = np.empty((len(names),len(data.time),len(data.x)*len(data.y)))
    for i, name in enumerate(names):
        X[i,:,:] = data[name].data.reshape(len(data.time), -1)
   
    gmed = geomedian.data.reshape((len(names),len(data.y)*len(data.x))) # geometric medians   
    cosdist = np.empty((len(data.time),len(data.y)*len(data.x))) # cosine distance
    cosdist.fill(np.nan)
    #CDistoutlier = np.zeros((len(data.y)*len(data.x)))
    for i in range(0,len(data.y)*len(data.x)):

        ind=[j for j in range(0,len(data.time)) if X[1,j,i]>0]
        if len(ind)>0:         
            cosdist[ind,i] = cos_distance(gmed[:,i], X[:,ind,i])
    #CDistoutlier = np.nanpercentile(cosdist,75,axis=0)+1.5*(np.nanpercentile(cosdist,75,axis=0)-np.nanpercentile(cosdist,25,axis=0))
    
    ds = xr.Dataset(coords={'time':data.time[:],'y':data.y[:],'x':data.x[:],'bands':names}, attrs={'crs':'EPSG:3577'})   
    ds['cosdist'] = (('time','y','x'),cosdist[:].reshape((len(data.time),len(data.y),len(data.x))).astype('float32'))
    #ds['CDistoutlier'] = (('y','x'),CDistoutlier.reshape((len(data.y),len(data.x))).astype('float32'))
    return ds

def nbrdist(data,geomedian):
    """
    Calculates the NBR (normalised burn ratio) and eculidean distance to the NBR from geometric median
    
    Args:
        data: (p x Y x X) matrix, where p = number of bands and Y x X is the size of the data
	    geomedian: (p x Y x X) matrix geometric median 
    Returns:
        ds:xarray with NBR, NBRdist, Change direction, outlier value
    """
    nir = data.nir.data.reshape((len(data.time),len(data.y)*len(data.x)))
    swir2 = data.swir2.data.reshape((len(data.time),len(data.y)*len(data.x)))
    
    NBRmed = np.zeros((len(data.y),len(data.x))) # NBR geometric median
    NBRmed = (geomedian.data[3,:,:]-geomedian.data[5,:,:])/(geomedian.data[3,:,:]+geomedian.data[5,:,:]) 
    
    NBRmed = NBRmed.reshape((len(data.y)*len(data.x)))
    NBR = np.empty((len(data.time),len(data.y)*len(data.x)))
    NBR.fill(np.nan)
    NBRdist = np.zeros((len(data.time),len(data.y)*len(data.x))) # NBR euclidean distance  
    direction = np.zeros((len(data.time),len(data.y)*len(data.x)),dtype='uint16') # NBR change direction
    #NBRoutlier = np.zeros((len(data.y)*len(data.x)))
    for i in range(0,len(data.y)*len(data.x)):    
        ind = np.where((nir[:,i])>0)[0]
        if len(ind)>0: 
            NBR[ind,i] = (nir[ind,i]-swir2[ind,i])/(nir[ind,i]+swir2[ind,i])
            NBRdist[ind,i],direction[ind,i] = nbr_eucdistance(NBRmed[i],NBR[ind,i]) 
            #NBRoutlier[i]=np.nanpercentile(NBRdist[ind,i],75,axis=0)+1.5*(np.nanpercentile(NBRdist[ind,i],75,axis=0)-np.nanpercentile(NBRdist[ind,i],25,axis=0))
    NBR = NBR.reshape((len(data.time),len(data.y),len(data.x)))
    NBRdist = NBRdist.reshape((len(data.time),len(data.y),len(data.x)))
    #NBRoutlier = NBRoutlier.reshape((len(data.y),len(data.x)))
    direction = direction.reshape((len(data.time),len(data.y),len(data.x)))
    ds = xr.Dataset(coords={'time':data.time[:],'y':data.y[:],'x':data.x[:]}, attrs={'crs':'EPSG:3577'})
    ds['NBR'] = (('time','y','x'),(NBR).astype('float32'))
    ds['NBRdist'] = (('time','y','x'),(NBRdist).astype('float32'))
    ds['ChangeDir'] = (('time','y','x'),direction)
    #ds['NBRoutlier'] = (('y','x'),(NBRoutlier).astype('float32'))
    
    return  ds

def region_growing(severity,ds):
    """
    
    """
    Start_Date=severity.StartDate.data[~np.isnan(severity.StartDate.data)].astype('<M8[ns]')
    ChangeDates=np.unique(Start_Date)
    i = 0
    sumpix = np.zeros(len(ChangeDates))
    for d in ChangeDates:
        Nd=np.sum(Start_Date==d)
        sumpix[i] = Nd    
        i = i+1
    ii = np.where(sumpix==np.max(sumpix))[0][0]
    z_distance=2/3 # times outlier distance (eq. 3 stdev)
    d=str(ChangeDates[ii])[:10]
    ti = np.where(ds.time>np.datetime64(d))[0][0]
    NBR_score=(ds.ChangeDir*ds.NBRDist)[ti,:,:]/ds.NBRoutlier
    cos_score=(ds.ChangeDir*ds.cosdist)[ti,:,:]/ds.CDistoutlier
    Potential=((NBR_score>z_distance)&(cos_score>z_distance)).astype(int)
    SeedMap=(severity.Severe>0).astype(int)
    SuperImp=Potential*SeedMap+Potential;
    from skimage import measure
    all_labels = measure.label(Potential.astype(int).values,background=0)
    #see http://www.scipy-lectures.org/packages/scikit-image/index.html#binary-segmentation-foreground-background
    #help(measure.label)
    NewPotential=0.*all_labels.astype(float) # replaces previous map "potential" with labelled regions
    for ri in range(1,np.max(np.unique(all_labels))): # ri=0 is the background, ignore that
        #print(ri)
        NewPotential[all_labels==ri]=np.mean(np.extract(all_labels==ri,SeedMap))

    # plot
    fraction_seedmap=0.25 # this much of region must already have been mapped as burnt to be included
    SeedMap=(severity.Severe.data>0).astype(int)
    AnnualMap=0.*all_labels.astype(float)
    ChangeDates=ChangeDates[sumpix>np.percentile(sumpix,60)]
    for d in ChangeDates:
        d=str(d)[:10]
        ti = np.where(ds.time>np.datetime64(d))[0][0]
        NBR_score=(ds.ChangeDir*ds.NBRDist)[ti,:,:]/ds.NBRoutlier
        cos_score=(ds.ChangeDir*ds.cosdist)[ti,:,:]/ds.CDistoutlier
        Potential=((NBR_score>z_distance)&(cos_score>z_distance)).astype(int)
        all_labels = measure.label(Potential.astype(int).values,background=0)
        NewPotential=0.*SeedMap.astype(float)
        for ri in range(1,np.max(np.unique(all_labels))): 
            NewPotential[all_labels==ri]=np.mean(np.extract(all_labels==ri,SeedMap))
        AnnualMap=AnnualMap+(NewPotential>fraction_seedmap).astype(int)
    BurnExtent=(AnnualMap>0).astype(int)
    BurnArea = BurnExtent*SeedMap+BurnExtent
    ba = xr.Dataset({'BurnArea':(('y','x'),BurnArea)},coords={'x':ds.x[:],'y':ds.y[:]})
    return ba

def severitymapping(data,period,method,growing=True):
    """
    Calculate burnt area with the given period
    Args:
        data: (t x Y x X) matrix, where t = number of days and Y x X is the size of the cosdist, nbrdist...
	    period: period of time with burn mapping interest,  e.g.('2015-01-01','2015-12-31') 
        method: methods for change detection
        growing: whether to grow the region 
    Returns:
        ds:xarray with detected burnt area, e.g. severe, medium
    """
    timeind = np.where((data.time<=np.datetime64(period[1]))&(data.time>=np.datetime64(period[0])))[0]
    data = data.sel(time=data.time[timeind])
    
    
    #if method==1:
    #    tmp = data.CosDist.where(data.CosDist>data.CDistoutlier).sum(axis=0).data
    #    tmp = tmp.reshape(((len(data.x)*len(data.y))))
    #    outlierind=np.where(tmp>0)[0]
    #    CDist=data.CosDist.data.reshape((len(data.time),len(data.x)*len(data.y)))[:,outlierind]
    #    CDistoutlier=data.CDistoutlier.data.reshape((len(data.x)*len(data.y)))[outlierind]
    #    ds = xr.Dataset(coords={'time':data.time[:],'points':np.linspace(0,len(outlierind)-1,len(outlierind)).astype('int32')})    
    #    ds['CosDist'] = (('time','points'),CDist)
        
    if method==2:
        tmp = data.cosdist.where((data.cosdist>data.CDistoutlier)&(data.NBR<0)).sum(axis=0).data
        tmp = tmp.reshape(((len(data.x)*len(data.y))))
        outlierind=np.where(tmp>0)[0]
        CDist=data.cosdist.data.reshape((len(data.time),len(data.x)*len(data.y)))[:,outlierind]
        CDistoutlier=data.CDistoutlier.data.reshape((len(data.x)*len(data.y)))[outlierind]
        NBR=data.NBR.data.reshape((len(data.time),len(data.x)*len(data.y)))[:,outlierind]
        NBRDist=data.NBRDist.data.reshape((len(data.time),len(data.x)*len(data.y)))[:,outlierind]
        NBRoutlier= data.NBRoutlier.data.reshape((len(data.x)*len(data.y)))[outlierind]
        ds = xr.Dataset(coords={'time':data.time[:],'points':np.linspace(0,len(outlierind)-1,len(outlierind)).astype('uint32')})    
        ds['NBR'] = (('time','points'),NBR)        
        ds['CosDist'] = (('time','points'),CDist)
        ds['CDistoutlier'] = (('points'),CDistoutlier)        
    elif method ==3:
        tmp = data.cosdist.where((data.cosdist>data.CDistoutlier)&(data.NBRDist>data.NBRoutlier)&(data.ChangeDir==1)).sum(axis=0).data
        tmp = tmp.reshape(((len(data.x)*len(data.y))))
        outlierind=np.where(tmp>0)[0]  
        CDist=data.cosdist.data.reshape((len(data.time),len(data.x)*len(data.y)))[:,outlierind]
        CDistoutlier=data.CDistoutlier.data.reshape((len(data.x)*len(data.y)))[outlierind]
        NBR=data.NBR.data.reshape((len(data.time),len(data.x)*len(data.y)))[:,outlierind]
        NBRDist=data.NBRDist.data.reshape((len(data.time),len(data.x)*len(data.y)))[:,outlierind]
        NBRoutlier= data.NBRoutlier.data.reshape((len(data.x)*len(data.y)))[outlierind]
        ChangeDir=data.ChangeDir.data.reshape((len(data.time),len(data.x)*len(data.y)))[:,outlierind] 
        ds = xr.Dataset(coords={'time':data.time[:],'points':np.linspace(0,len(outlierind)-1,len(outlierind)).astype('uint32')})    
        ds['NBRDist'] = (('time','points'),NBRDist)
        ds['CosDist'] = (('time','points'),CDist)
        ds['ChangeDir'] = (('time','points'),ChangeDir)
        ds['CDistoutlier'] = (('points'),CDistoutlier)
        ds['NBRoutlier'] = (('points'),NBRoutlier)
    else:
        raise ValueError  
    print(len(outlierind))
    sev = np.zeros((len(outlierind)))
    dates = np.zeros((len(outlierind)))
    days = np.zeros((len(outlierind)))
    for i in range(0,len(outlierind)): 
        
        sev[i], dates[i], days[i]=severity(time=data.time,data=ds.sel(points=i),method=method)
        
    sevindex=np.zeros((len(data.x)*len(data.y)))
    duration=np.zeros((len(data.x)*len(data.y)))
    startdate=np.zeros((len(data.x)*len(data.y)))
    
    sevindex[outlierind]=sev
    duration[outlierind]=days
    startdate[outlierind]=dates
    
    sevindex=sevindex.reshape((len(data.y),len(data.x)))
    duration=duration.reshape((len(data.y),len(data.x)))
    startdate=startdate.reshape((len(data.y),len(data.x)))
    #startdate[startdate==0]=np.nan
    out = xr.Dataset(coords={'y':data.y[:],'x':data.x[:]},attrs={'crs':'EPSG:3577'})
    
    out['StartDate']=(('y','x'),startdate)
    out['Duration']=(('y','x'),duration.astype('uint16'))
    burnt = np.zeros((len(data.y),len(data.x)))
    burnt[duration>1] = 1
    out['Severe']=(('y','x'),burnt.astype('uint16'))
    if growing==True:
        BurnArea = region_growing(out,data)
        out['Medium'] = (('y','x'),BurnArea.BurnArea.data.astype(int)) 

        
    
    return out
        
        
   