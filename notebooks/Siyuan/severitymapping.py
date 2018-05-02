
"""
This is a script to calculate severity from the Cosine Distance.
Required inputs:
CDist: cosine distance in 3D as [time,y,x]
Time: time
Outputs: sevindex in [y,x] as the area below curve, startdate and duration
"""
import xarray as xr
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def severitymapping(CDist,Time):
    #calculate the outliers as ( 75th percentile + 1.5*IQR) for each location and output as 2D array [y,x]
    outlier = np.nanpercentile(CDist,75,axis=0)+1.5*(np.nanpercentile(CDist,75,axis=0)-np.nanpercentile(CDist,25,axis=0))


    sevindex = np.zeros(((CDist.shape[1]),(CDist.shape[2])))
    Duration = np.zeros(((CDist.shape[1]),(CDist.shape[2])))
    startdate = np.zeros(((CDist.shape[1]),(CDist.shape[2])))
    for x in range(0,(CDist.shape[2])):
        for y in range(0,(CDist.shape[1])):
                
            notnanind = np.where(~np.isnan(CDist[:,y,x]))[0] #remove the nan values for each pixel
            cosdist = CDist[notnanind,y,x]
            outlierind = np.where(cosdist>outlier[y,x])[0]
            time = Time[notnanind]
            outlierdates = time[outlierind]
            Nout = len(outlierind)
            AreaAboveD0 = 0
            if Nout>=2:
                tt = []            
                for ii in range(0,Nout):
                    if outlierind[ii]+1<len(time):
                        u = np.where(time[outlierind[ii]+1]==outlierdates)[0] 
                        #print(u)

                        if len(u)>0:
                            t1_t0 = (time[outlierind[ii]+1]-time[outlierind[ii]])/np.timedelta64(1, 's')/(60*60*24)
                            y1_y0 = (cosdist[outlierind[ii]+1] +cosdist[outlierind[ii]] )-2*outlier[y,x]
                            AreaAboveD0 = AreaAboveD0 + 0.5*y1_y0*t1_t0
                            Duration[y,x] = Duration[y,x] + t1_t0
                            tt.append(ii) # record the index where it is detected as a change

                if len(tt)>0:                
                    startdate[y,x] = time[outlierind[tt[0]]] #record the date of the first change 
                    sevindex[y,x] = AreaAboveD0


    #ds = xr.Dataset({'Severity':(('y','x'),sevindex[:,:])},coords={'y':Northing[:],'x':Easting[:]})
    #ds.to_netcdf(outputfilename)
    return sevindex, startdate, Duration

