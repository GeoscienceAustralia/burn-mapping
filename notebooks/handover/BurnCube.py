import datacube as dc
from datacube.helpers import ga_pq_fuser
from datacube.storage import masking
import numpy as np
import xarray as xr
import multiprocessing as mp
import ctypes
from contextlib import closing
import datetime
import warnings
from stats import nbr_eucdistance, cos_distance, severity, outline_to_mask, hotspot_polygon, nanpercentile, post_filtering
from shapely import geometry


FASTGM=False
try:
    from pcm import gmpcm as geometric_median
    print("PCM geomedian loaded")
    FASTGM = True
except ImportError:
    from stats import geometric_median
warnings.filterwarnings('ignore')


def _create_geospatial_attributes(dataset):
    import pyproj
    y_max, y_min, x_max, x_min = np.max(dataset.y),np.min(dataset.y),np.max(dataset.x),np.min(dataset.x)
    # convert to projected centre coordinates
    wgs84 = pyproj.Proj(init='epsg:4326')
    gda94 = pyproj.Proj(init='epsg:3577')
    [lon_max,lon_min],[lat_max, lat_min] = pyproj.transform(gda94,wgs84,[x_max,x_min],[y_max,y_min])
    #print([lon_max,lon_min],[lat_max, lat_min])
                             
    p1 = geometry.Point(lon_min,lat_min)
    p2 = geometry.Point(lon_max,lat_min)
    p3 = geometry.Point(lon_max,lat_max)
    p4 = geometry.Point(lon_min,lat_max)

    pointList = [p1, p2, p3, p4, p1]
    poly = geometry.Polygon([[p.x, p.y] for p in pointList])
    return lat_max, lat_min, lon_max, lon_min, poly


def _create_variable_attributes(dataset):
    dataset['y'].attrs={'units':'metre', 'long_name':'y coordinate of projection','standard_name': 'projection_y_coordinate'}
    dataset['x'].attrs={'units':'metre', 'long_name':'x coordinate of projection','standard_name': 'projection_x_coordinate'}
    dataset['StartDate'].attrs={'long_name':'StartDate','standard_name':'StartDate','axis':'T','coverage_content_type':'model results'}
    dataset['Duration'].attrs={'units':'number of days', 'long_name':'Duration','standard_name':'Duration','coverage_content_type':'model results'}
    dataset['Severity'].attrs={'units':'temporal integral of cosine distance for the duration of detected severe burned', 'long_name':'Severity','standard_name':'Severity','coverage_content_type':'model results'}
    dataset['Severe'].attrs={'units':'1', 'long_name':'Severe burned area','standard_name':'Severe','coverage_content_type':'model results'}
    dataset['Moderate'].attrs={'units':'1', 'long_name':'Moderate burned area','standard_name':'Moderate','coverage_content_type':'model results'}
    dataset['Corroborate'].attrs={'units':'1', 'long_name':'Corrorborate evidence','standard_name':'Corroborate','coverage_content_type':'model results'}
    dataset['Cleaned'].attrs={'units':'month', 'long_name':'Cleaned','standard_name':'Cleaned','coverage_content_type':'model results'}
    return dataset


def create_attributes(dataset,product_name,version, method, res=25):
    dataset = _create_variable_attributes(dataset)
    lat_max, lat_min, lon_max, lon_min, poly = _create_geospatial_attributes(dataset)
    product_definition = {
    'name': product_name,
    'description': 'Description for ' + product_name,
    'mapping_Method': method, # NBR or NBRdist
    'data_source':'Landsat 5/8',
    'resolution':str(res)+' m',    
    'cdm_data_type': 'Grid',
    'cmi_id': 'BAM_25_1.0',
    'Conventions': 'CF-1.6, ACDD-1.3',
    'geospatial_bounds': str(poly),
    'geospatial_bounds_crs':'EPSG:4326',
    'geospatial_lat_max': lat_max,
    'geospatial_lat_min': lat_min,
    'geospatial_lat_units': 'degree_north',
    'geospatial_lon_max': lon_max,
    'geospatial_lon_min': lon_min,
    'geospatial_lon_units': 'degree_east',
    'date_created':  datetime.datetime.today().isoformat(),
    'history': "NetCDF-CF file created by datacube version '1.6rc2+108.g096a26d5' at 20180914",
    'institution': "Commonwealth of Australia (Geoscience Australia)",
    'keywords': "AU/GA,NASA/GSFC/SED/ESD/LANDSAT,ETM+,TM,OLI,EARTH SCIENCE, BURNED AREA",
    'keywords_vocabulary': "GCMD",
    "license": "CC BY Attribution 4.0 International License",
    "product_version": version,
    "publisher_email": "earth.observation@ga.gov.au",
    "publisher_url": "http://www.ga.gov.au",
    "references": "Renzullo et al (2019)",
    "source": "Burned Area Map Algorithm v1.0",
    'summary': "Burned area is mapped from Landsat surface reflectance time series. It implements a change detection algorithm developed jointly by ANU and GA. The layers available are: StartDate, detected start-date of severe and moderate burned area; Duration, duration of land cover change due to the bushfire; Severity, severity of land cover change due to the bushfire; Severe, binary mask for severe burnt area; Moderate, binary mask for moderate and severe burnt area; Corroborate, binary mask for corroborating evidence from hotspots data with 4km buffer; Cleaned, month of first detection, filtered by spatial and temporal coincidence with corroborate layer",
    "title":"Burned Area Map Annual 1.0",
    "crs_wkt":'PROJCS["GDA94 / Australian Albers",GEOGCS["GDA94",DATUM["D_GDA_1994",SPHEROID["GRS_1980",6378137,298.257222101]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]],PROJECTION["Albers"],PARAMETER["standard_parallel_1",-18],PARAMETER["standard_parallel_2",-36],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",132],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["Meter",1]]',
               
    }
    for key, value in product_definition.items():
        dataset.attrs[key] = value
    
    return dataset

def dist_geomedian(params):
    """
    multiprocess version with shared memory of the geomedian
    """
    gmed = np.frombuffer(shared_out_arr.get_obj(), dtype=np.float32).reshape((params[2][0], params[2][2]))
    X = np.frombuffer(shared_in_arr.get_obj(), dtype=np.int16).reshape(params[2])

    for i in range(params[0], params[1]):
        ind = np.where(X[1, :, i] > 0)[0]
        if len(ind) > 0:
            gmed[:, i] = geometric_median(X[:, ind, i], params[3], params[4])
            

def dist_distance(params):
    """
    multiprocess version with shared memory of the cosine distances and nbr distances
    """
    X = np.frombuffer(shared_in_arr1.get_obj(), dtype=np.int16).reshape(params[2])
    gmed = np.frombuffer(shared_in_arr2.get_obj(), dtype=np.float32).reshape((params[2][0], params[2][2]))
    cos_dist = np.frombuffer(shared_out_arr1.get_obj(), dtype=np.float32).reshape((params[2][1], params[2][2]))
    nbr_dist = np.frombuffer(shared_out_arr2.get_obj(), dtype=np.float32).reshape((params[2][1], params[2][2]))
    direction = np.frombuffer(shared_out_arr3.get_obj(), dtype=np.int16).reshape((params[2][1], params[2][2]))

    for i in range(params[0], params[1]):
        ind = np.where(X[1, :, i] > 0)[0]

        if len(ind) > 0:
            cos_dist[ind, i] = cos_distance(gmed[:, i], X[:, ind, i])
            nbrmed = (gmed[3, i] - gmed[5, i]) / (gmed[3, i] + gmed[5, i])
            nbr = (X[3, :, i] - X[5, :, i]) / (X[3, :, i] + X[5, :, i])
            nbr_dist[ind, i], direction[ind, i] = nbr_eucdistance(nbrmed, nbr[ind])
        

def dist_severity(params):
    """
    multiprocess version with shared memory of the severity algorithm
    """
    NBR = np.frombuffer(shared_in_arr01.get_obj(), dtype=np.float32).reshape((-1, params[2]))
    NBRDist = np.frombuffer(shared_in_arr02.get_obj(), dtype=np.float32).reshape((-1, params[2]))
    CDist = np.frombuffer(shared_in_arr03.get_obj(), dtype=np.float32).reshape((-1, params[2]))
    ChangeDir = np.frombuffer(shared_in_arr04.get_obj(), dtype=np.int16).reshape((-1, params[2]))
    NBRoutlier = np.frombuffer(shared_in_arr05.get_obj(), dtype=np.float32)
    CDistoutlier = np.frombuffer(shared_in_arr06.get_obj(), dtype=np.float32)
    t = np.frombuffer(shared_in_arr07.get_obj(), dtype=np.float64)
   
    sev = np.frombuffer(shared_out_arr01.get_obj(), dtype=np.float64)
    dates = np.frombuffer(shared_out_arr02.get_obj(), dtype=np.float64)
    days = np.frombuffer(shared_out_arr03.get_obj(), dtype=np.float64)
    for i in range(params[0], params[1]):
        
        sev[i], dates[i], days[i] = severity(NBR[:, i], NBRDist[:, i], CDist[:, i], ChangeDir[:, i],
                                             NBRoutlier[i], CDistoutlier[i],t, method=params[3])
       
        

class BurnCube(dc.Datacube):
    def __init__(self):
        super(BurnCube, self).__init__(app='TreeMapping.getLandsatStack')

        self.band_names = ['red', 'green', 'blue', 'nir', 'swir1', 'swir2']
        self.dataset = None
        self.geomed = None
        self.dists = None
        self.outlrs = None
        

    def to_netcdf(self, path):
        """Saves input data to a netCDF4 on disk
            Args:
                path string: path to a file on disk
        """
        self.dataset.to_netcdf(path)

    def open_dataset(self, path):
        """Loads input data from a netCDF4 on disk
            Args:
                path string: path to a file on disk
        """
        self.dataset = xr.open_dataset(path)

    def geomed_to_netcdf(self, path):
        """Saves geomedian data to a netCDF4 on disk
            Args:
                path string: path to a file on disk
        """
        self.geomed.to_netcdf(path)

    def open_geomed(self, path):
        """Loads geomedian data from a netCDF4 on disk
            Args:
                path string: path to a file on disk
        """
        self.geomed = xr.open_dataset(path)

    def _load_pq(self, x, y, res, period, n_landsat):
        query = {
            'time': period,
            'x': x,
            'y': y,
            'crs': 'EPSG:3577',
            'measurements': ['pixelquality'],
            'resolution': res,
        }

        pq_stack = []
        for n in n_landsat:
            pq_stack.append(self.load(product='ls{}_pq_albers'.format(n),
                                      group_by='solar_day', fuse_func=ga_pq_fuser,
                                      resampling='nearest', **query))

        pq_stack = xr.concat(pq_stack, dim='time').sortby('time')

        # Land/sea mask isn't used at the moment. Possible alternatives are WOFS and ITEM.
        #pq_stack['land'] = masking.make_mask(pq_stack.pixelquality, land_sea='land')
        # masking cloud, saturation and invalid data (contiguous)

        pq_stack['land'] = masking.make_mask(pq_stack.pixelquality, land_sea='land')
        pq_stack['no_cloud'] = masking.make_mask(pq_stack.pixelquality, cloud_acca='no_cloud',
                                                 cloud_fmask='no_cloud', cloud_shadow_acca='no_cloud_shadow',
                                                 cloud_shadow_fmask='no_cloud_shadow',blue_saturated=False,
                                                 green_saturated=False,
                                                 red_saturated=False,
                                                 nir_saturated=False,
                                                 swir1_saturated=False,
                                                 swir2_saturated=False,
                                                 contiguous=True)


        return pq_stack

    def _load_nbart(self, x, y, res, period, n_landsat):
        query = {
            'time': period,
            'x': x,
            'y': y,
            'crs': 'EPSG:3577',
            'measurements': self.band_names,
            'resolution': res,
        }

        nbart_stack = []
        for n in n_landsat:
            dss = self.find_datasets(product='ls{}_nbart_albers'.format(n), **query)
            nbart_stack.append(self.load(product='ls{}_nbart_albers'.format(n),
                                         group_by='solar_day', datasets=dss, resampling='bilinear',
                                         **query))

        nbart_stack = xr.concat(nbart_stack, dim='time').sortby('time')

        return nbart_stack

    def load_cube(self, x, y, res, period, n_landsat):
        """Loads the Landsat data for the selected region and sensors
            Note:
                This method loads the data into the self.dataset variable.
            Args:
                x list float: horizontal min max range
                y list float: vertical min max range
                res float: pixel resolution for the input data
                period list datetime: temporal range for input data
                n_landsat int: number of the Landsat mission
        """
        
        nbart_stack = self._load_nbart(x, y, res, period, n_landsat)
        pq_stack = self._load_pq(x, y, res, period, n_landsat)
        pq_stack, nbart_stack = xr.align(pq_stack, nbart_stack, join='inner')
        #mask = np.nanmean(pq_stack.good_pixel.values.reshape(pq_stack.good_pixel.shape[0], -1), axis=1) > .2
        #X = nbart_stack.sel(time=mask).where(pq_stack.good_pixel.sel(time=mask), 0, drop=False) # keep data as integer

        #self.dataset = X.to_array(dim='band').to_dataset(dim='cube') 
        pq_stack['good_pixel'] = pq_stack.no_cloud.where(nbart_stack.red > 0, False, drop=False)
        pq_stack['no_bright_pixel'] = pq_stack.no_cloud.where(nbart_stack.blue <900, False, drop=False)
        goodpix = pq_stack.no_cloud * (pq_stack.pixelquality > 0) * pq_stack.good_pixel *pq_stack.no_bright_pixel
        mask = np.nanmean(goodpix.values.reshape(goodpix.shape[0], -1), axis=1) > .2
        cubes = [nbart_stack[band][mask, :, :] * goodpix[mask, :, :] for band in self.band_names]
        X = np.stack(cubes, axis=0)

        data = xr.Dataset(coords={'band': self.band_names,
                                  'time': nbart_stack.time[mask],
                                  'y': nbart_stack.y[:],
                                  'x': nbart_stack.x[:]})
        data["cube"] = (('band', 'time', 'y', 'x'), X)
        data.time.attrs = []

        self.dataset = data

        
    def geomedian(self, period, **kwargs):
        if FASTGM:
            self.fastgeomedian(period)
        else:
            self.hdgeomedian(period, **kwargs)
        
    def fastgeomedian(self, period):
        """
        Calculates the geometric median of band reflectances using pcm.
        """
        
        _X = self.dataset['cube'].sel(time=slice(period[0], period[1]))
        t_dim = _X.time[:]
        X = _X.transpose('y','x','band','time').data.astype(np.float32)
        X[X==0] = np.nan
        gmed = geometric_median(X)
        
        ds = xr.Dataset(coords={'time': t_dim, 'y': self.dataset.y[:], 'x': self.dataset.x[:],
                                'band': self.dataset.band}, attrs={'crs': 'EPSG:3577'})
        ds['geomedian'] = (('band', 'y', 'x'), gmed.transpose(2,0,1))
        
        del gmed, X
        self.geomed = ds
        
    def hdgeomedian(self, period, n_procs=4, epsilon=.5, max_iter=40):
        """
        Calculates the geometric median of band reflectances in parallel
        The procedure stops when either the error 'epsilon' or the maximum
        number of iterations 'max_iter' is reached.
            Note:
                This method saves the result of the computation into the
                self.geomed variable: p-dimensional vector with geometric
                median reflectances, where p is the number of bands.
            Args:
                period: range of dates to compute the geomedian
                n_procs: tolerance criterion to stop iteration
                epsilon: tolerance criterion to stop iteration
                max_iter: maximum number of iterations
        """
        
        n = len(self.dataset.y) * len(self.dataset.x)
        out_arr = mp.Array(ctypes.c_float, len(self.dataset.band) * n)
        gmed = np.frombuffer(out_arr.get_obj(), dtype=np.float32).reshape((len(self.dataset.band), n))
        gmed.fill(np.nan)

        _X = self.dataset['cube'].sel(time=slice(period[0], period[1]))
        t_dim =_X.time[:] 
        
        if len(t_dim)<6:
            print("no enough data for the calculation of geomedian")
            return

        in_arr = mp.Array(ctypes.c_short, len(self.dataset.band) * len(_X.time) * n)
        X = np.frombuffer(in_arr.get_obj(), dtype=np.int16).reshape((len(self.dataset.band), len(_X.time), n))
        X[:] = _X.data.reshape((len(self.dataset.band), len(_X.time), -1))

        def init(shared_in_arr_, shared_out_arr_):
            global shared_in_arr
            global shared_out_arr
            shared_in_arr = shared_in_arr_
            shared_out_arr = shared_out_arr_

        with closing(mp.Pool(initializer=init, initargs=(in_arr, out_arr,))) as p:
            chunk = n // n_procs
            p.map_async(dist_geomedian, [(i, min(n, i + chunk), X.shape, epsilon, max_iter) for i in range(0, n, chunk)])
        p.join()

        ds = xr.Dataset(coords={'time': t_dim, 'y': self.dataset.y[:], 'x': self.dataset.x[:],
                                'band': self.dataset.band}, attrs={'crs': 'EPSG:3577'})
        ds['geomedian'] = (('band', 'y', 'x'), gmed[:].reshape((len(self.dataset.band), len(self.dataset.y),
                                                                 len(self.dataset.x))).astype(np.float32))

        del gmed, in_arr, out_arr, X
        self.geomed = ds

    def distances(self, period, n_procs=4):
        """
        Calculates the cosine distance between observation and reference.
        The calculation is point based, easily adaptable to any dimension.
            Note:
                This method saves the result of the computation into the
                self.dists variable: p-dimensional vector with geometric
                median reflectances, where p is the number of bands.
            Args:
                period: range of dates to compute the geomedian
                n_procs: tolerance criterion to stop iteration
        """
        
        n = len(self.dataset.y) * len(self.dataset.x)
        _X = self.dataset['cube'].sel(time=slice(period[0], period[1]))

        t_dim = _X.time.data
        if len(t_dim)<1:
            print("---%d observations" %(len(t_dim)))
            self.dists = None
            print("no enough data for the calculation of distances")
            return
        nir = _X[3, :, :, :].data.astype('float32')
        swir2 = _X[5, :, :, :].data.astype('float32')
        nir[nir <= 0] = np.nan
        swir2[swir2 <= 0] = np.nan
        nbr = ((nir - swir2) / (nir + swir2))

        out_arr1 = mp.Array(ctypes.c_float, len(t_dim) * n)
        out_arr2 = mp.Array(ctypes.c_float, len(t_dim) * n)
        out_arr3 = mp.Array(ctypes.c_short, len(t_dim) * n)

        cos_dist = np.frombuffer(out_arr1.get_obj(), dtype=np.float32).reshape((len(t_dim), n))
        cos_dist.fill(np.nan)
        nbr_dist = np.frombuffer(out_arr2.get_obj(), dtype=np.float32).reshape((len(t_dim), n))
        nbr_dist.fill(np.nan)
        direction = np.frombuffer(out_arr3.get_obj(), dtype=np.int16).reshape((len(t_dim), n))
        direction.fill(0)
        
        in_arr1 = mp.Array(ctypes.c_short, len(self.dataset.band) * len(_X.time) * n)
        X = np.frombuffer(in_arr1.get_obj(), dtype=np.int16).reshape((len(self.dataset.band), len(_X.time), n))
        X[:] = _X.data.reshape(len(self.dataset.band), len(_X.time), -1)

        in_arr2 = mp.Array(ctypes.c_float, len(self.dataset.band) * n)
        gmed = np.frombuffer(in_arr2.get_obj(), dtype=np.float32).reshape((len(self.dataset.band), n))
        gmed[:] = self.geomed['geomedian'].data.reshape(len(self.dataset.band), -1)

        def init(shared_in_arr1_, shared_in_arr2_, shared_out_arr1_, shared_out_arr2_, shared_out_arr3_):
            global shared_in_arr1
            global shared_in_arr2
            global shared_out_arr1
            global shared_out_arr2
            global shared_out_arr3

            shared_in_arr1 = shared_in_arr1_
            shared_in_arr2 = shared_in_arr2_
            shared_out_arr1 = shared_out_arr1_
            shared_out_arr2 = shared_out_arr2_
            shared_out_arr3 = shared_out_arr3_

        with closing(mp.Pool(initializer=init, initargs=(in_arr1, in_arr2, out_arr1, out_arr2, out_arr3,))) as p:
            chunk = n // n_procs
            if n==0 or chunk==0:
                return
            p.map_async(dist_distance, [(i, min(n, i + chunk), X.shape) for i in range(0, n, chunk)])

        p.join()

        ds = xr.Dataset(coords={'time': t_dim, 'y': self.dataset.y[:], 'x': self.dataset.x[:],
                                'band': self.dataset.band}, attrs={'crs': 'EPSG:3577'})
        ds['cosdist'] = (('time', 'y', 'x'), cos_dist[:].reshape((len(t_dim), len(self.dataset.y),
                                                                  len(self.dataset.x))).astype('float32'))
        ds['NBRDist'] = (('time', 'y', 'x'), nbr_dist[:].reshape((len(t_dim), len(self.dataset.y),
                                                                  len(self.dataset.x))).astype('float32'))
        ds['ChangeDir'] = (('time', 'y', 'x'), direction[:].reshape((len(t_dim), len(self.dataset.y),
                                                                     len(self.dataset.x))).astype('float32'))
        ds['NBR'] = (('time', 'y', 'x'), nbr)
        del in_arr1, in_arr2, out_arr1, out_arr2, out_arr3, gmed, X, cos_dist, nbr_dist, direction, nbr

        self.dists = ds

    def outliers(self):
        """
        Calculate the outliers for distances for change detection 
        """
        #print(self.dists)
        if self.dists is None:
            print("no distances for the outlier calculations")
            return
        NBRps = nanpercentile(self.dists.NBRDist.data, [25,75])
        NBRoutlier = NBRps[1] + 1.5 * (NBRps[1]-NBRps[0])
        CDistps=nanpercentile(self.dists.cosdist.data, [25,75])
        CDistoutlier = CDistps[1] +1.5 * (CDistps[1]-CDistps[0])
        
        ds = xr.Dataset(coords={'y': self.dataset.y[:], 'x': self.dataset.x[:]}, attrs={'crs': 'EPSG:3577'})
        ds['CDistoutlier'] = (('y', 'x'), CDistoutlier.astype('float32'))
        ds['NBRoutlier'] = (('y', 'x'), NBRoutlier.astype('float32'))
        self.outlrs = ds

    def region_growing(self, severity):
        """
        The function includes further areas that do not qualify as outliers but do show a substantial decrease in NBR and 
        are adjoining pixels detected as burns. These pixels are classified as 'moderate severity burns'.
            Note: this function is build inside the 'severity' function
                Args:
                    severity: xarray including severity and start-date of the fire
        """
        Start_Date = severity.StartDate.data[~np.isnan(severity.StartDate.data)].astype('datetime64[ns]')
        ChangeDates = np.unique(Start_Date)
        i = 0
        sumpix = np.zeros(len(ChangeDates))
        # calculate the number of burnt pixels in each date
        for d in ChangeDates:
            Nd = np.sum(Start_Date == d)
            sumpix[i] = Nd
            i = i + 1
            
        z_distance = 2 / 3  # times outlier distance (eq. 3 stdev)
       
        from skimage import measure       
        # see http://www.scipy-lectures.org/packages/scikit-image/index.html#binary-segmentation-foreground-background      
        fraction_seedmap = 0.25  # this much of region must already have been mapped as burnt to be included
        SeedMap = (severity.Severe.data > 0).astype(int) #use 'Severe' burns as seed map to grow
        ChangeDates = ChangeDates[sumpix > np.percentile(sumpix, 50)] # only grow the region with burnt pixel more than the median 
        Startdates = np.zeros((len(self.dists.y),len(self.dists.x)))
        Startdates[~np.isnan(severity.StartDate.data)] = Start_Date 
        tmpMap = SeedMap 
        AnnualMap = SeedMap
        # grow the region based on StartDate
        for d in ChangeDates:
         
            di = str(d)[:10]
            ti = np.where(self.dists.time > np.datetime64(di))[0][0]
            NBR_score = (self.dists.ChangeDir * self.dists.NBRDist)[ti, :, :] / self.outlrs.NBRoutlier
            cos_score = (self.dists.ChangeDir * self.dists.cosdist)[ti, :, :] / self.outlrs.CDistoutlier
            Potential = ((NBR_score > z_distance) & (cos_score > z_distance)).astype(int)
            #Use the following line if using NBR is preferred
            #Potential = ((self.dists.NBR[ti, :, :] > 0) & (cos_score > z_distance)).astype(int) 
            
            all_labels = measure.label(Potential.astype(int).values, background=0) # labelled all the conneted component 
            NewPotential = 0. * SeedMap.astype(float)
            for ri in range(1, np.max(np.unique(all_labels))):
                NewPotential[all_labels == ri] = np.mean(np.extract(all_labels == ri, SeedMap))
            
            AnnualMap = AnnualMap + (NewPotential > fraction_seedmap).astype(int)          
            tmp = (AnnualMap > 0).astype(int)
            Startdates[(tmp-tmpMap)>0] = d # assign the same date to the growth region
            tmpMap=tmp

        BurnExtent = (AnnualMap > 0).astype(int)
        #Startdates[BurnExtent==0]=0
        
        return BurnExtent, Startdates


    def severitymapping(self, period, n_procs=4, method='NBR', growing=True):
        """Calculates burnt area for a given period
            Args:
                period: period of time with burn mapping interest,  e.g.('2015-01-01','2015-12-31')
                n_procs: tolerance criterion to stop iteration
                method: methods for change detection
                growing: whether to grow the region
        """
        if self.dists is None:
            print("No data available for severity mapping")
            return None
        #data = self.datasets.sel(time=slice(period[0], period[1]))
        CDist = self.dists.cosdist.data.reshape((len(self.dists.time), -1))
        CDistoutlier = self.outlrs.CDistoutlier.data.reshape((len(self.dists.x) * len(self.dists.y)))
        NBRDist = self.dists.NBRDist.data.reshape((len(self.dists.time), -1))
        NBR = self.dists.NBR.data.reshape((len(self.dists.time), -1))
        NBRoutlier = self.outlrs.NBRoutlier.data.reshape((len(self.dists.x) * len(self.dists.y)))
        ChangeDir = self.dists.ChangeDir.data.reshape((len(self.dists.time), -1))

        if method == 'NBR':
            tmp = self.dists.cosdist.where((self.dists.cosdist > self.outlrs.CDistoutlier) &
                                           (self.dists.NBR < 0)).sum(axis=0).data
            tmp = tmp.reshape((len(self.dists.x) * len(self.dists.y)))
            outlierind = np.where(tmp > 0)[0]
           

        elif method == 'NBRdist':
            tmp = self.dists.cosdist.where((self.dists.cosdist > self.outlrs.CDistoutlier) &
                                           (self.dists.NBRDist > self.outlrs.NBRoutlier) &
                                           (self.dists.ChangeDir == 1)).sum(axis=0).data
            tmp = tmp.reshape((len(self.dists.x) * len(self.dists.y)))
            outlierind = np.where(tmp > 0)[0]         

        else:
            raise ValueError

        if len(outlierind)==0:
            print('no burnt area detected')
            return None
        #input shared arrays
        in_arr1 = mp.Array(ctypes.c_float, len(self.dists.time[:])*len(outlierind))
        NBR_shared = np.frombuffer(in_arr1.get_obj(), dtype=np.float32).reshape((len(self.dists.time[:]), len(outlierind)))
        NBR_shared[:] = NBR[:, outlierind]

        in_arr2 = mp.Array(ctypes.c_float, len(self.dists.time[:])*len(outlierind))
        NBRDist_shared = np.frombuffer(in_arr2.get_obj(), dtype=np.float32).reshape((len(self.dists.time[:]), len(outlierind)))
        NBRDist_shared[:] = NBRDist[:, outlierind]

        in_arr3 = mp.Array(ctypes.c_float, len(self.dists.time[:])*len(outlierind))
        CosDist_shared = np.frombuffer(in_arr3.get_obj(), dtype=np.float32).reshape((len(self.dists.time[:]), len(outlierind)))
        CosDist_shared[:] = CDist[:, outlierind]

        in_arr4 = mp.Array(ctypes.c_short, len(self.dists.time[:])*len(outlierind))
        ChangeDir_shared = np.frombuffer(in_arr4.get_obj(), dtype=np.int16).reshape((len(self.dists.time[:]), len(outlierind)))
        ChangeDir_shared[:] = ChangeDir[:, outlierind]

        in_arr5 = mp.Array(ctypes.c_float, len(outlierind))
        NBRoutlier_shared = np.frombuffer(in_arr5.get_obj(), dtype=np.float32)
        NBRoutlier_shared[:] = NBRoutlier[outlierind]

        in_arr6 = mp.Array(ctypes.c_float, len(outlierind))
        CDistoutlier_shared = np.frombuffer(in_arr6.get_obj(), dtype=np.float32)
        CDistoutlier_shared[:] = CDistoutlier[outlierind]

        in_arr7 = mp.Array(ctypes.c_double, len(self.dists.time[:]))
        t = np.frombuffer(in_arr7.get_obj(), dtype=np.float64)
        t[:] = self.dists.time.data.astype('float64')
       
        #output shared arrays
        out_arr1 = mp.Array(ctypes.c_double, len(outlierind))
        sev = np.frombuffer(out_arr1.get_obj(), dtype=np.float64)
        sev.fill(np.nan)

        out_arr2 = mp.Array(ctypes.c_double, len(outlierind))
        dates = np.frombuffer(out_arr2.get_obj(), dtype=np.float64)
        dates.fill(np.nan)

        out_arr3 = mp.Array(ctypes.c_double, len(outlierind))
        days = np.frombuffer(out_arr3.get_obj(), dtype=np.float64)
        days.fill(0)

        def init(shared_in_arr1_, shared_in_arr2_, shared_in_arr3_, shared_in_arr4_, shared_in_arr5_,
                 shared_in_arr6_, shared_in_arr7_,shared_out_arr1_, shared_out_arr2_, shared_out_arr3_):
            global shared_in_arr01
            global shared_in_arr02
            global shared_in_arr03
            global shared_in_arr04
            global shared_in_arr05
            global shared_in_arr06
            global shared_in_arr07
            global shared_out_arr01
            global shared_out_arr02
            global shared_out_arr03

            shared_in_arr01 = shared_in_arr1_
            shared_in_arr02 = shared_in_arr2_
            shared_in_arr03 = shared_in_arr3_
            shared_in_arr04 = shared_in_arr4_
            shared_in_arr05 = shared_in_arr5_
            shared_in_arr06 = shared_in_arr6_
            shared_in_arr07 = shared_in_arr7_
            shared_out_arr01 = shared_out_arr1_
            shared_out_arr02 = shared_out_arr2_
            shared_out_arr03 = shared_out_arr3_

        
        with closing(mp.Pool(initializer=init, initargs=(in_arr1, in_arr2, in_arr3, in_arr4, in_arr5, in_arr6,in_arr7,
                                                         out_arr1, out_arr2, out_arr3,))) as p:
            chunk = len(outlierind) // n_procs
            if len(outlierind)==0 or chunk==0:
                print("no burnt pixel")
                return
            p.map_async(dist_severity, [(i, min(len(outlierind), i + chunk), len(outlierind),method) for i in range(0, len(outlierind), chunk)])
        
        p.join()
    
        sevindex = np.zeros((len(self.dists.y)*len(self.dists.x)))
        duration = np.zeros((len(self.dists.y)*len(self.dists.x)))*np.nan
        startdate = np.zeros((len(self.dists.y)*len(self.dists.x)))*np.nan
        sevindex[outlierind] = sev
        duration[outlierind] = days        
        startdate[outlierind] = dates
        sevindex = sevindex.reshape((len(self.dists.y), len(self.dists.x)))
        duration = duration.reshape((len(self.dists.y), len(self.dists.x)))
        startdate = startdate.reshape((len(self.dists.y), len(self.dists.x)))
        startdate[startdate==0] = np.nan
        duration[duration==0] = np.nan
        del in_arr1, in_arr2, in_arr3, in_arr4, in_arr5, in_arr6,out_arr1, out_arr2, out_arr3
        del sev,days,dates,NBR_shared,NBRDist_shared,CosDist_shared,NBRoutlier_shared,ChangeDir_shared,CDistoutlier_shared
        
        out = xr.Dataset(coords={'y': self.dists.y[:], 'x': self.dists.x[:]})
        out['StartDate'] = (('y', 'x'), startdate)
        out['Duration'] = (('y', 'x'), duration.astype('int16'))
        burnt = np.zeros((len(self.dists.y), len(self.dists.x)))
        burnt[duration > 1] = 1       
        out['Severity']=(('y','x'),sevindex.astype('float32'))
        out['Severe'] = (('y', 'x'), burnt.astype('int16'))
        
        if burnt.sum() == 0:
            out['Corroborate'] = (('y', 'x'), np.zeros((len(self.dists.y),len(self.dists.x))).astype('int16'))
            out['Moderate'] = (('y', 'x'), np.zeros((len(self.dists.y),len(self.dists.x))).astype('int16'))
            out['Cleaned'] = (('y', 'x'), np.zeros((len(self.dists.y),len(self.dists.x))).astype('int16'))
            return create_attributes(out,'Burned Area Map','v1.0', method)

        if growing == True:
            BurnArea,growing_dates = self.region_growing(out)
            out['Moderate'] = (('y', 'x'), BurnArea.astype('int16'))
            growing_dates[growing_dates==0] = np.nan
            out['StartDate'] = (('y', 'x'), growing_dates)
        
        extent = [np.min(self.dists.x.data), np.max(self.dists.x.data),
                  np.min(self.dists.y.data), np.max(self.dists.y.data)]
       
        #find the startdate for the fire and extract hotspots data
        values, counts = np.unique(startdate, return_counts=True)
        firedate = values[counts==np.max(counts)]
        startdate = (firedate.astype('datetime64[ns]')-np.datetime64(1, 'M')).astype('datetime64[ns]')
        stopdate = (firedate.astype('datetime64[ns]')-np.datetime64(-1, 'M')).astype('datetime64[ns]')
        #fireperiod = (str(startdate)[2:12],str(stopdate)[2:12])
        fireperiod = period
        #print(fireperiod)
        polygons = hotspot_polygon(fireperiod, extent, 4000)  # generate hotspot polygons with 4km buffer

        #default mask
        HotspotMask=np.zeros((len(self.dists.y),len(self.dists.x)))

        if polygons==None :
            print('No hotspots data.')
        elif polygons.is_empty:
            print('No hotspots data.')
        else:
            coords = out.coords

            if polygons.type == 'MultiPolygon':
                for polygon in polygons:
                    HotspotMask_tmp = outline_to_mask(polygon.exterior, coords['x'], coords['y'])
                    HotspotMask = HotspotMask_tmp + HotspotMask
                HotspotMask=xr.DataArray(HotspotMask, coords=coords, dims=('y', 'x'))
            if polygons.type == 'Polygon':
                HotspotMask = outline_to_mask(polygons.exterior, coords['x'], coords['y'])
                HotspotMask = xr.DataArray(HotspotMask, coords=coords, dims=('y', 'x'))
 
        out['Corroborate'] = (('y', 'x'), HotspotMask.astype('int16'))
        out = post_filtering(out,hotspots_filtering=True,date_filtering=True)
        return create_attributes(out,'Burned Area Map','v1.0', method)


