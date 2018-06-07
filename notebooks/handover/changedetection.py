"""
"""
from burnmappingtoolbox import severity
import pandas as pd
import numpy as np
import xarray as xr
def severity_mapping(data,period,method):
    sevindex = np.zeros((len(data.y),len(data.x)))
    duration = np.zeros((len(data.y),len(data.x)))
    startdate = np.zeros((len(data.y),len(data.x)))
    timeind = timeind=np.where((data.time<=np.datetime64(period[1]))&(data.time>=np.datetime64(period[0])))
    for x in range(0,len(data.x)):
        for y in range(0,len(data.y)):
            sevindex[y,x], startdate[y,x], duration[y,x]=severity(CDist=data.CosDist.data[:,y,x],CDistoutlier=data.CDistoutlier.data[y,x],Time=data.time,
                                                                  NBR=data.NBR.data[:,y,x],NBRDist=data.NBRDist.data[:,y,x],NBRoutlier=data.NBRoutlier.data[y,x], Sign=data.NegtiveChange.data[:,y,x], Method=method)
    
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
    
    return firedataset,ds
