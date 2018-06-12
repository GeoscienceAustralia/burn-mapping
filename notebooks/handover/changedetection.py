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
    sevindex = np.zeros((len(data.y),len(data.x)))
    duration = np.zeros((len(data.y),len(data.x)))
    startdate = np.zeros((len(data.y),len(data.x)))
    timeind = np.where((data.time<=np.datetime64(period[1]))&(data.time>=np.datetime64(period[0])))[0]
    print("There were {0} observations from {1} to {2}.".format(str(len(timeind)), period[0], period[1]))
    for x in range(0,len(data.x)):
        for y in range(0,len(data.y)):
            sevindex[y,x], startdate[y,x], duration[y,x]=severity(CDist=data.CosDist.data[timeind,y,x],CDistoutlier=data.CDistoutlier.data[y,x],Time=data.time[timeind],
                                                                  NBR=data.NBR.data[timeind,y,x],NBRDist=data.NBRDist.data[timeind,y,x],NBRoutlier=data.NBRoutlier.data[y,x], Sign=data.NegtiveChange.data[timeind,y,x], Method=method)
    
    ds = xr.Dataset({'Severity':(('y','x'),sevindex)},coords={'y':data.y[:],'x':data.x[:]},attrs={'crs':'EPSG:3577'})
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


    
