"""
This script calulate the severity for the given region and return the start-date, duration and the severity plot
Required inputs:
data: the data array from the geomedian_and_dists module, including the cosine distance, NBRdist, outliers, NBR 
period: the period for severity calculation e.g.('2015-07-01','2016-06-30')
plot: True for severity plot, False for none plot

Outputs:
firedataset: dataset including the locations, severity, start date, duration information for all the burned pixels 
ds: severity information in 2D
"""
from burnmappingtoolbox import severity
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib
def severity_mapping(data,period,method,plot=True):
    
    sevindex = np.zeros((len(data.y),len(data.x))).astype('f4')
    duration = np.zeros((len(data.y),len(data.x))).astype('f4')
    startdate = np.zeros((len(data.y),len(data.x))).astype('f4')
    timeind = np.where((data.time<=np.datetime64(period[1]))&(data.time>=np.datetime64(period[0])))[0]
    print("There were {0} observations from {1} to {2}.".format(str(len(timeind)), period[0], period[1]))
    for x in range(0,len(data.x)):
        for y in range(0,len(data.y)):
            sevindex[y,x], startdate[y,x], duration[y,x]=severity(CDist=data.CosDist.data[timeind,y,x],CDistoutlier=data.CDistoutlier.data[y,x],Time=data.time[timeind],
                                                                  NBR=data.NBR.data[timeind,y,x],NBRDist=data.NBRDist.data[timeind,y,x],NBRoutlier=data.NBRoutlier.data[y,x], Sign=data.NegtiveChange.data[timeind,y,x], Method=method)
    
    ds = xr.Dataset({'Severity':(('y','x'),sevindex),'StartDate':(('y','x'),startdate)},coords={'y':data.y[:].astype('f4'),'x':data.x[:].astype('f4')},attrs={'crs':'EPSG:3577'})
    startdate[startdate==0]=np.nan
    row,col = np.where(~np.isnan(startdate))
    tmpdate = startdate.ravel()
    Start_Date = startdate[row,col].astype('<M8[ns]')
    Easting = data.x[col]
    Northing = data.y[row]
    Severity = sevindex[row,col]
    Duration = duration[row,col]
    firedataset=pd.DataFrame({'Easting':Easting,'Northing':Northing,'Start-Date':Start_Date,'Severity':Severity,'Duration':Duration})
    
    colors = pd.read_fwf('mycolormap.txt',header=None)
    if plot==True:
        if len(firedataset.Severity)>0:
            font = {'family' : 'normal','weight' : 'bold', 'size'   : 16}

            matplotlib.rc('font', **font)
            colors = pd.read_fwf('mycolormap.txt',header=None)
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10,10))
            mycolormap = np.zeros((len(colors),3))
            mycolormap[:,0]=colors[0]
            mycolormap[:,1]=colors[1]
            mycolormap[:,2]=colors[2]
            cmap =matplotlib.colors.LinearSegmentedColormap.from_list('my_colormap',mycolormap/256)
            plt.figure(figsize=(7,7))
            maxsev=np.percentile(firedataset.Severity,98)
            im=plt.imshow(sevindex,cmap=cmap,vmax=maxsev)
            plt.colorbar(im,fraction=0.03, pad=0.05)
            im.axes.get_xaxis().set_visible(False)
            im.axes.get_yaxis().set_visible(False)
            plt.title(period[0]+' to '+period[1])
        else:
            print("No change found from {0} to {1}.".format( period[0], period[1]))
    return firedataset,ds


def region_growing(severity,ds):
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
    NBR_score=(ds.NegtiveChange*ds.NBRDist)[ti,:,:]/ds.NBRoutlier
    cos_score=(ds.NegtiveChange*ds.CosDist)[ti,:,:]/ds.CDistoutlier
    Potential=((NBR_score>z_distance)&(cos_score>z_distance)).astype(int)
    SeedMap=(severity.Severity>0).astype(int)
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
    SeedMap=(severity.Severity.data>0).astype(int)
    AnnualMap=0.*all_labels.astype(float)
    ChangeDates=ChangeDates[sumpix>np.percentile(sumpix,60)]
    for d in ChangeDates:
        d=str(d)[:10]
        ti = np.where(ds.time>np.datetime64(d))[0][0]
        NBR_score=(ds.NegtiveChange*ds.NBRDist)[ti,:,:]/ds.NBRoutlier
        cos_score=(ds.NegtiveChange*ds.CosDist)[ti,:,:]/ds.CDistoutlier
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


    
