import numpy as np
import datetime as datetime

def geometric_median(x, epsilon=1, max_iter=40):
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
    y0 = np.nanmean(x, axis=1)
    if len(y0[np.isnan(y0)]) > 0:
        return y0

    for _ in range(max_iter):
        euc_dist = np.transpose(np.transpose(x) - y0)
        euc_norm = np.sqrt(np.sum(euc_dist ** 2, axis=0))
        not_nan = np.where(~np.isnan(euc_norm))[0]
        y1 = np.sum(x[:, not_nan] / euc_norm[not_nan], axis=1) / (np.sum(1 / euc_norm[not_nan]))
        if len(y1[np.isnan(y1)]) > 0 or np.sqrt(np.sum((y1 - y0) ** 2)) < epsilon:
            return y1

        y0 = y1

    return y0


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
    ref = ref.astype('float32')
    obs = obs.astype('float32')
    cosdist = np.empty((obs.shape[1],))
    cosdist.fill(np.nan)
    index = np.where(~np.isnan(obs[0, :]))[0]
    cosdist[index] = np.transpose(
        [1 - np.sum(ref * obs[:, t]) / (np.sqrt(np.sum(ref ** 2)) * np.sqrt(np.sum(obs[:, t] ** 2))) for t in index])

    return cosdist


def nbr_eucdistance(ref, obs):
    """
    Returns the euclidean distance between the NBR at each time step with the NBR calculated from the geometric medians
    and also the direction of change to the NBR from the geometric medians.
    
    Args:
        ref: NBR calculated from geometric median, one value
        obs: NBR time series, 1-D time series array with ndays 
    
    Returns:
        nbr_dist: the euclidean distance 
        direction: change direction (1: decrease; 0: increase) at each time step in [ndays]
    """
    nbr_dist = np.empty((obs.shape[0],))
    direction = np.zeros((obs.shape[0],), dtype='uint8')
    nbr_dist.fill(np.nan)
    index = np.where(~np.isnan(obs))[0]
    euc_dist = (obs[index] - ref)
    euc_norm = np.sqrt((euc_dist ** 2))
    nbr_dist[index] = euc_norm
    direction[index[euc_dist < -0.05]] = 1

    return nbr_dist, direction


def severity(NBR, NBRDist, CDist, ChangeDir, NBRoutlier, CDistoutlier, t,method='NBRdist'):
    """
    Returns the severity,duration and start date of the change.
    Args:

        time: dates of observations
        data: xarray including the cosine distances, NBR distances, NBR, change direction and outliers value
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
    
    notnanind = np.where(~np.isnan(CDist))[0]  # remove the nan values for each pixel

    if method == 'NBR':  # cosdist above the line and NBR<0
        outlierind = np.where((CDist[notnanind] > CDistoutlier) & (NBR[notnanind] < 0))[0]
        cosdist = CDist[notnanind]

    elif method == 'NBRdist':  # both cosdist and NBR dist above the line and it is negative change
        outlierind = np.where((CDist[notnanind] > CDistoutlier) &
                              (NBRDist[notnanind] > NBRoutlier) &
                              (ChangeDir[notnanind] == 1))[0]

        cosdist = CDist[notnanind]
    else:
        raise ValueError
    t = t.astype('datetime64[ns]')
    t = t[notnanind]
    outlierdates = t[outlierind]
    n_out = len(outlierind)
    area_above_d0 = 0
    if n_out >= 2:
        tt = []
        for ii in range(0, n_out):
            if outlierind[ii] + 1 < len(t):
                u = np.where(t[outlierind[ii] + 1] == outlierdates)[0]  # next day have to be outlier to be included
                # print(u)

                if len(u) > 0:
                    t1_t0 = (t[outlierind[ii] + 1] - t[outlierind[ii]]) / np.timedelta64(1, 's') / (60 * 60 * 24)
                    y1_y0 = (cosdist[outlierind[ii] + 1] + cosdist[outlierind[ii]]) - 2 * CDistoutlier
                    area_above_d0 = area_above_d0 + 0.5 * y1_y0 * t1_t0  # calculate the area under the curve
                    duration = duration + t1_t0
                    tt.append(ii)  # record the index where it is detected as a change

        if len(tt) > 0:
            startdate = t[outlierind[tt[0]]]  # record the date of the first change
            sevindex = area_above_d0
 

    return sevindex, startdate, duration


def outline_to_mask(line, x, y):
    """Create mask from outline contour

    Parameters
    ----------
    line: array-like (N, 2)
    x, y: 1-D grid coordinates (input for meshgrid)

    Returns
    -------
    mask : 2-D boolean array (True inside)

    Examples
    --------
    >>> from shapely.geometry import Point
    >>> poly = Point(0, 0).buffer(1)
    >>> x = np.linspace(-5, 5, 100)
    >>> y = np.linspace(-5, 5, 100)
    >>> mask = outline_to_mask(poly.boundary, x, y)
    """
    import matplotlib.path as mplp
    mpath = mplp.Path(line)
    X, Y = np.meshgrid(x, y)
    points = np.array((X.flatten(), Y.flatten())).T
    mask = mpath.contains_points(points).reshape(X.shape)

    return mask


def hotspot_polygon(year, extent, buffersize):
    """Create polygons for the hotspot with a buffer
    year: given year for hotspots data
    extent: [xmin,xmax,ymin,ymax] in crs EPSG:3577
    buffersize: in meters
    
    Examples:
    ------------
    >>>year=2017
    >>>extent = [1648837.5, 1675812.5, -3671837.5, -3640887.5]
    >>>polygons = hotspot_polygon(year,extent,4000)
    """
    import glob
    import pyproj
    import pandas as pd
    datafile = '/g/data/xc0/original/GA_SentinelHotspots/hotspot_historic_*.csv'
    gda94aa = pyproj.Proj(init='epsg:3577')
    gda94 = pyproj.Proj(init='epsg:4283')

    if year == 2005:
        name = '/g/data/xc0/original/GA_SentinelHotspots/hotspot_historic_2005-2010.csv'
        table = pd.read_csv(name)

    elif year == 2010:
        name = '/g/data/xc0/original/GA_SentinelHotspots/hotspot_historic_2010-2015.csv'
        table = pd.read_csv(name)

    else:
        for i in range(0, len(glob.glob(datafile))):
            name = glob.glob(datafile)[i]
            startyear = int(name[-13: -9])
            endyear = int(name[-8: -4])
            if (year <= endyear) & (year >= startyear):
                table = pd.read_csv(name)
                break
            if year >= 2018:
                print("No hotspot data")        
                return None
    
    start = np.datetime64(datetime.datetime(year, 1, 1))
    stop = np.datetime64(datetime.datetime(year, 12, 31))

    dates = table.datetime.values.astype('datetime64')
    lon, lat = pyproj.transform(gda94aa, gda94, extent[0:2], extent[2:4])
    index = np.where((dates >= start) * (dates <= stop) * (table.latitude <= lat[1]) *
                     (table.latitude >= lat[0]) * (table.longitude <= lon[1]) *
                     (table.longitude >= lon[0]))[0]
    latitude = table.latitude.values[index]
    longitude = table.longitude.values[index]
    easting, northing = pyproj.transform(gda94, gda94aa, longitude, latitude)

    from shapely.ops import cascaded_union
    from shapely.geometry import Point

    patch = [Point(easting[i], northing[i]).buffer(buffersize) for i in range(0, len(index))]
    polygons = cascaded_union(patch)

    return polygons

