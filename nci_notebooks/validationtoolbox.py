"""
This script includes several modules for the validation purpose, including treecover_masking, validation polygon masking,
roc_analysis...
----------------------------
The available modules are listed below:
    burnpixel_masking(data): for masking burned pixel
    treecover_masking(year,data,prctg=60): for masking forest and none forest pixel
    validation_dataset_config(State,Validation_period,BurnPixel): for validation dataset configuration
    CreateValidatedBurnMask(BurnPixel,State, Validation_period): for validation mask generation
    validate(Test_Array = None, Validated_Array = None, plot=False): for roc analysis 
    
    _forward_fill(Input_DataArray = None): for missing value filling
    _identify_burned_area(Input_DataArray,lag0_threshold = -0.50, lag1_threshold = -0.3): for NBR differencing method
    transform_from_latlon(lat, lon): for rasterising the shapefile
    rasterize(shapes, coords, fill=np.nan, **kwargs): rasterize a list of (geometry, fill_value) tuples
    
"""

import xarray as xr
import numpy as np
import pandas as pd
import geopandas
from rasterio import features
from affine import Affine
import matplotlib.pyplot as plt
from pylab import rcParams
pd.set_option('display.max_colwidth', 200)
pd.set_option('display.max_rows', None)
rcParams['figure.figsize'] = (12,8)
import warnings
import datetime as datetime
warnings.filterwarnings("ignore")

def burnpixel_masking(data,varname):
    """
    This function converts the severity map into a burn pixel mask
    Required input:
    data: severity data in 2D, e.g. output from severity_mapping in the changedection.py
    """
    Burnpixel = data[varname]
    Burnpixel.data[Burnpixel.data>1]=1
    return Burnpixel

def treecover_masking(year,data,prctg=60,size='25m'):
    """
    return the treecover mask for the given region in epsg:3577
    the threshold for the forest mask can be changed, the default value is 60%
    Required inputs:
    year: the tree cover in which year for masking, e.g. one year before the fire 
    data: dataset with x and y coordinates
    
    Outputs:
    mask: including both forest and noneforest mask
    """
    import pyproj
    gda94aa = pyproj.Proj(init='epsg:3577')#,py_list='aea')
    gda94 = pyproj.Proj(init='epsg:4326')
    lon1,lat1=pyproj.transform(gda94aa,gda94,data.x.data[0],data.y.data[0])
    lon2,lat2=pyproj.transform(gda94aa,gda94,data.x.data[-1],data.y.data[-1])
    #lon1,lat1=(data.x.data[0],data.y.data[0])
    #lon2,lat2=(data.x.data[-1],data.y.data[-1])
    
    if size=='25m':
        filename = 'http://dapds00.nci.org.au/thredds/dodsC/ub8/au/treecover/ANUWALD.TreeCover.25m.%s.nc' %str(year)
    else:
        filename = 'http://dapds00.nci.org.au/thredds/dodsC/ub8/au/treecover/250m/ANUWALD.TreeCover.%s.250m.nc' %str(year)
    #filename = '/g/data/u46/users/ekb100/burn-mapping/validation/ANUWALD.TreeCover.25m.%s.nc' %str(year)

    
    TC = xr.open_dataset(filename)
    yr = str(year)+'-12-31'
    TC = TC.TreeCover.sel(time=yr)
    lonmin = min([lon1,lon2])
    latmin = min([lat1,lat2])
    lonmax = max([lon1,lon2])
    latmax = max([lat1,lat2])
    # extract the tree cover for the given region but 0.1 further in each direction for resample purpose
    
    row = np.where((TC.latitude.data>latmin-0.1)&(TC.latitude.data<latmax+0.1))[0] 
    col = np.where((TC.longitude.data>lonmin-0.1)&(TC.longitude.data<lonmax+0.1))[0]
    longitude, latitude = np.meshgrid(TC.longitude.data[col], TC.latitude.data[row])
    
    easting,northing=pyproj.transform(gda94,gda94aa,longitude.ravel(),latitude.ravel()) # change the projection
    Treecover = TC.squeeze()[col,row].transpose()
    x, y = np.meshgrid(data.x.data, data.y.data)
    from scipy.interpolate import griddata
    gridnew = griddata((easting,northing),Treecover.data.ravel(),(x,y),method='nearest',fill_value=np.nan) # resample to the given data resolution and extent
    ds = xr.Dataset({'Treecover':(('y','x'),gridnew)},coords={'y':data.y,'x':data.x})
    TreecoverMask = np.zeros((gridnew.shape))
    TreecoverMask[:] = gridnew
    if size=='25m':
        TreecoverMask[TreecoverMask<=0]=0 # forest mask
        TreecoverMask[TreecoverMask>0]=1
        TreecoverMask[np.isnan(TreecoverMask)]=0
    else:
        TreecoverMask[TreecoverMask<=prctg]=0 # forest mask
        TreecoverMask[TreecoverMask>prctg]=1
    TreecoverMask2 = np.zeros((gridnew.shape))
    TreecoverMask2[:] = gridnew
    if size=='25m':
        TreecoverMask2[TreecoverMask2<=0]=1 # none forest mask
        TreecoverMask2[TreecoverMask2>0]=0
        TreecoverMask2[np.isnan(TreecoverMask2)]=1
    else:
        TreecoverMask2[TreecoverMask2<=prctg]=1 # none forest mask
        TreecoverMask2[TreecoverMask2>prctg]=0
    mask = xr.Dataset({'ForestMask':(('y','x'),TreecoverMask),'NoneForestMask':(('y','x'),TreecoverMask2)},coords={'y':data.y,'x':data.x})
    return mask

def validation_dataset_config(State,Validation_period,BurnPixel):
    """
    Please note: the working directory of the fire perimeters polygons are currently set to /g/data/xc0/projectBurn_Mapping/02_Fire_Perimeters_Polygons/
    please modify if needed.
    You may also need to modify the file paths, file names and potentially the column names as well to 
    match the data that you have.
    
    This function sets the correct validation shapefile for validation and densifies the data to the validation period and 
    the geographic extent of the burnpixel mask
    
    Required inputs:
    State: state/territory name abbreviation, e.g. TAS, VIC, NSW, ACT..
    Validation_period: e.g.('2015-01-01','2015-12-31')
    BurnPixel: burn pixel generated by the "burnpixel_masking" with x,y coordinates
    
    Outputs:
    df: dataset including the polygons
    date variable
    """
    #workingdir = '/g/data/xc0/project/Burn_Mapping/02_Fire_Perimeters_Polygons/'
    workingdir = '/g/data/u46/users/ekb100/burn-mapping/validation/'
    
    if State =='TAS':
        #shapefile_filepath = workingdir+'TAS_2017_State_Fire_History/fire_history_all_fires_20170907.shp'
        shapefile_filepath = workingdir+'tas/TAS_2017_State_Fire_History/fire_history_all_fires_20170907.shp'
        df = geopandas.read_file(shapefile_filepath)[['geometry','IGN_DATE']]
        df.columns=['geometry','Burn_Date']

    if State =='VIC':
        #shapefile_filepath = workingdir+'Victoria_FIRE_HISTORY/FIRE_HISTORY.shp'
        #shapefile_filepath = workingdir+'VIC/FIRE_HISTORY.shp'
        shapefile_filepath = workingdir+'new_vic/ll_gda94/sde_shape/whole/VIC/FIRE/layer/fire_history_lastburnt.shp'
        df = geopandas.read_file(shapefile_filepath)[['geometry','START_DATE']]
        df.columns=['geometry','Burn_Date']
        
    if State =='NSW':
        #shapefile_filepath = workingdir+'NSW_RFS_Outlines/NSW_Fire_History/WildFireHistory.shp'
        shapefile_filepath = workingdir+'nsw/firehistoryp/FireHistory_P.shp'
        #df = geopandas.read_file(shapefile_filepath)[['geometry','ENDDATE']]
        df = geopandas.read_file(shapefile_filepath)[['geometry','EndDate']]
        df.columns=['geometry','Burn_Date']
        
    if State =='SA':
        #shapefile_filepath = workingdir+'SA/FIREMGT_FireHistory_shp/FIREMGT_FireHistory.shp'
        shapefile_filepath = workingdir+'sa/FIREMGT_FireHistory_shp/FIREMGT_FireHistory.shp'
        df = geopandas.read_file(shapefile_filepath)[['geometry','FIREDATE']]
        df.columns=['geometry','Burn_Date']     
    
    if State =='ACT':
        #shapefile_filepath = workingdir+'FireHistory_ACT/ShapeFile/FireHistory.shp'
        shapefile_filepath = workingdir+'ACT/ShapeFile/FireHistory.shp'
        df = geopandas.read_file(shapefile_filepath)[['geometry','DATE']]
        df.columns=['geometry','Burn_Date']
       
    if State =='QLD':
        #shapefile_filepath = workingdir+'QLD/osmfirewildfire1992to2015/OSM_FIRE_WILDFIRE_1992to2015.shp'
        shapefile_filepath = workingdir+'ql/osmfirewildfire1992to2015.shp/OSM_FIRE_WILDFIRE_1992to2015.shp'
        df1 = geopandas.read_file(shapefile_filepath)[['geometry','YEAR_BURN_']]
        df1.columns = ['geometry','Burn_Date']
        df1.Burn_Date=pd.to_datetime(df1.Burn_Date,format='%Y').astype('datetime64[ns]').astype('str')

        #shapefile_filepath = workingdir+'QLD/plannedburnshistory20092017/Planned_Burns_History_2009_2017.shp'
        #df2 = geopandas.read_file(shapefile_filepath)[['geometry','Date_Compl']] 
        #df2.columns = ['geometry','Burn_Date']

        #shapefile_filepath = workingdir+'QLD/wildfirereport2011to2018/WildfireReport_2011to2018.shp'
        shapefile_filepath = workingdir+'ql/wildfirereport2011to2018.shp/WildfireReport_2011to2018.shp'
        df3 = geopandas.read_file(shapefile_filepath)[['geometry','DateKNown']]  
        df3.columns = ['geometry','Burn_Date']

        df=pd.concat([df1, df3])
        
    df = df[(df.Burn_Date>=Validation_period[0])&(df.Burn_Date<=Validation_period[1])]
    df = df.to_crs({'init': 'epsg:3577'})
    x_extent = [BurnPixel.x.min() , BurnPixel.x.max()]
    y_extent = [BurnPixel.y.min() , BurnPixel.y.max()]
    df = df.cx[ x_extent[0]:x_extent[1] , y_extent[0]:y_extent[1]]
    return df,df.Burn_Date

#roc analysis
def _forward_fill(Input_DataArray = None):
    """
    - backfills all NAN values with the last non-nan value in the xr.DataArray
    - this step is necessary, because the differencing technique gives misleading results when NAN values are present

    INPUT:
    - Input_DataArray must be a xarray DataArray
    """
    temp_array = Input_DataArray
    
    for i in np.arange(0,len(temp_array.time)):
        temp_array = temp_array.fillna(temp_array.shift(time=1))
    
    return(temp_array)

def _identify_burned_area(Input_DataArray,lag0_threshold = -0.50, lag1_threshold = -0.3):
    """
    - calculates dNBR (difference in NBR between consecutive non-NAN values)
    - creates a burn mask (0: unburned, 1: burned) which satisfies two conditions:
        i) lag0_threshold: dNBR between time = t and time = t-1 exceeds this threshold
            - this catches all pixels which experience a sufficiently large drop in NBR 
            
        ii) lag1_threshold: dNBR between time = t+1 and time = t-1 exceeds this threshold
            - this catches all pixels in which a sufficiently large drop in NBR persists for at least one 
            additional timestep (this attempts to filter out erronous single values) 
    """
    
    A = _forward_fill(Input_DataArray)
    B1 = A - A.shift(time=1)
    B2 = A.shift(time=-1)-A.shift(time=1)
    
    C = B1.where(B1 < lag0_threshold)
    D = (B1 < lag0_threshold) & (B2 < lag1_threshold)
    
    return D

def transform_from_latlon(lat, lon):
    """
    This function is used for rasterising the shapefile
    """
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    trans = Affine.translation(lon[0], lat[0])
    scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
    return trans * scale

def rasterize(shapes, coords, fill=np.nan, **kwargs):
    """Rasterize a list of (geometry, fill_value) tuples onto the given
    xray coordinates. This only works for 1d latitude and longitude arrays.
    """
    transform = transform_from_latlon(coords['y'], coords['x'])
    out_shape = (len(coords['y']), len(coords['x']))
    raster = features.rasterize(shapes, out_shape=out_shape, transform = transform,
                                fill=fill, all_touched = True,
                                dtype=float, **kwargs)
    
    return xr.DataArray(raster, coords=coords, dims=('y', 'x'))

def CreateValidatedBurnMask(BurnPixel,State, Validation_period ):
    """
    This function generates the validation mask for the given region
    """  
    assert type(BurnPixel) is xr.core.dataarray.DataArray, 'Test_Array not an xarray DataArray'     
    df,StartDate=validation_dataset_config(State,Validation_period,BurnPixel)
                 
    coords = BurnPixel.coords
    if len(np.unique(StartDate))==0:
        print("No validation data available")
        return ([])
    else:
        for date in np.unique(StartDate):
            shapes = [(shape,1) for  n, shape in enumerate(df[StartDate == date].geometry)]

            try:
                new_da = rasterize(shapes , coords)
                new_da = new_da.assign_coords(time = date)
                new_da = new_da.expand_dims('time')
                output_array = xr.concat((output_array,new_da), dim='time')

            except NameError:
                output_array = rasterize(shapes , coords)
                output_array = output_array.assign_coords(time = date)
                output_array = output_array.expand_dims('time')


        return(output_array)

def validate(Test_Array = None, Validated_Array = None, plot=False):
    """
    This function validates the Test_Array against the Validated_Array.
    It then calculates the desired validation metrics.
    
    To create ROC Curves, run this function multiple times, iterating over the range of threshold values. 
        
    ##############    
    Inputs:
    ----------------------
        Test_Array: is an xarray DataArray with values 0 and 1 
        Validated_Array: is an xarray DataArray with values 0 and 1 
    
        Plot = True/False : if True, output includes a plot showing Correct/Omission/Comission
    Outputs:
        [FPR,TPR] for false positive rate and true positive rate
    
    """
    
    import xarray
    assert type(Test_Array) is xarray.core.dataarray.DataArray, 'Test_Array not an xarray DataArray'
    assert type(Validated_Array) is xarray.core.dataarray.DataArray, 'Validated_Array not an xarray DataArray'

    # Classify as Correct/Omission/Comission
    Correct = Validated_Array.where(Validated_Array==1).where(Test_Array==1)
    Correct = Correct.fillna(0)
    Omission = Validated_Array.where(Validated_Array==1).where(Test_Array==0)
    Omission = Omission.fillna(0)
    Commission = Validated_Array.where(Validated_Array==0).where(Test_Array==1)+1
    Commission = Commission.fillna(0)
    
    Combined = Correct + Omission + Commission
    Combined = Combined.where(Combined!=0)
    # Calculate number of pixels in each category    
    e11 = Correct.sum(('x','y')).values                       # True Positives
    e12 = Commission.sum(('x','y')).values                    # False Positives
    e21 = Omission.sum(('x','y')).values                      # False Negatives
    e22 = Validated_Array.count().values - e11 - e12 - e21    # True Positives

    # Checksum (all values must be positive)
    assert e11.all() >= 0, 'e11 error metrics invalid' 
    assert e12.all() >= 0, 'e12 error metrics invalid' 
    assert e21.all() >= 0, 'e21 error metrics invalid' 
    assert e22.all() >= 0, 'e22 error metrics invalid' 
    
    commission_error_ratio = e12/(e11+e12)
    omission_error_ratio = e21/(e11+e21)
    dice_coeff = (2*e11)/(2*e11+e12+e21)
    bias = e12 - e21
    
    #
    TPR = e11/(e11+e21) # True Positive Ratio
    FPR = e12/(e12+e22) # False Positive Ratio
    
    metrics = {'commission_error_ratio': commission_error_ratio,
               'omission_error_ratio': omission_error_ratio,
               'dice_coeff': dice_coeff,
               'bias': bias
              }
    Correct = Validated_Array.where(Validated_Array==1).where(Test_Array==1)+2
    Correct = Correct.fillna(0)
    Omission = Validated_Array.where(Validated_Array==1).where(Test_Array==0)+1
    Omission = Omission.fillna(0)
    Commission = Validated_Array.where(Validated_Array==0).where(Test_Array==1)+1
    Commission = Commission.fillna(0)

    Combined = Correct + Omission + Commission
    Combined = Combined.where(Combined.values!=0)
       
    # FOR PLOTTING ONLY
    if plot==True:
	
        

        from matplotlib.colors import ListedColormap
	
        fig,ax = plt.subplots()
        cMap = ListedColormap(['white','sienna', 'sandybrown','darkgreen'])
        cax=Combined.plot(ax=ax,levels=[0,1, 2, 3, 4], cmap=cMap,add_colorbar=False)
        cbar = fig.colorbar(cax,ticks=[0.5,1.5,2.5,3.5])
        cbar.ax.set_yticklabels(['correct unburned area','false burned area','missed burned area','correct burned area'])
        cax.axes.get_xaxis().set_visible(False)
        cax.axes.get_yaxis().set_visible(False)
    return([FPR,TPR],Combined)

def validate_forest_grass(Test_Array = None, Validated_Array = None,Mask = None, plot=False):
    
    Test_forest = Test_Array*Mask.ForestMask
    Test_grass = Test_Array*Mask.NoneForestMask
    ForestMask = Validated_Array*Mask.ForestMask
    NoneForestMask = Validated_Array*Mask.NoneForestMask
    
    forest=Mask.ForestMask>0
    nonforest=Mask.NoneForestMask>0
    Tree,Combined1=validate(Test_Array = Test_forest.where(forest, drop=True), 
                            Validated_Array = ForestMask.where(forest, drop=True), plot=False)
        
    Grass,Combined2=validate(Test_Array = Test_grass.where(nonforest, drop=True), 
                             Validated_Array = NoneForestMask.where(nonforest, drop=True), plot=False)
    import matplotlib
    font = {'family' : 'normal',
            'weight' : 'normal',
            'size'   : 16}

    matplotlib.rc('font', **font)
    if plot==True:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        fig,axes = plt.subplots()
        cMap = ListedColormap(['white','darkred', 'peru','darkgreen'])
        cax=Combined1.plot(levels=[0,1, 2, 3, 4], cmap=cMap,add_colorbar=False)

        axes.patch.set_facecolor('none')
        cMap2 = ListedColormap(['white','indianred', 'burlywood','g'])
        Combined2.plot(ax=axes,levels=[0,1, 2, 3, 4], cmap=cMap2,add_colorbar=False,alpha=0.9)
        
        ds = xr.Dataset(coords={ 'y': Test_Array.y[:], 'x': Test_Array.x[:]}, attrs={'crs': 'EPSG:3577'})
        ds['tmp'] = (( 'y', 'x'), np.random.uniform(low=0, high=7, size=(len(Test_Array.y),len(Test_Array.x))))
        cMap3 = ListedColormap(['white','indianred','darkred',  'burlywood','peru','g','darkgreen'])
        cax2=ds.tmp.plot(ax=axes,levels=[0,1, 2, 3, 4, 5, 6, 7], cmap=cMap3,add_colorbar=False,alpha=1)
        cax2.set_visible(False)
        cbar = fig.colorbar(cax2,ticks=[0.5,1.5,2.5,3.5,4.5,5.5,6.5])
        cbar.ax.set_yticklabels(['correct unburned area','false burned grassland area','false burned forest area','missed burned grassland area','missed burned forest area','correct burned grassland area','correct burned forest area'])

        cax.axes.get_xaxis().set_visible(False)
        cax.axes.get_yaxis().set_visible(False)
    return Tree, Grass

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
    >>> poly = Point(0,0).buffer(1)
    >>> x = np.linspace(-5,5,100)
    >>> y = np.linspace(-5,5,100)
    >>> mask = outline_to_mask(poly.boundary, x, y)
    """
    import matplotlib.path as mplp
    mpath = mplp.Path(line)
    X, Y = np.meshgrid(x, y)
    points = np.array((X.flatten(), Y.flatten())).T
    mask = mpath.contains_points(points).reshape(X.shape)
    return mask

