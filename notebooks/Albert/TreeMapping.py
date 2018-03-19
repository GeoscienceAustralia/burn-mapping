
# coding: utf-8

# In[352]:


# Maps forest cover at 1/4000th degree using machine learning (random forests)
# a spatial subset of NCAS tree mapping is the predictand 
# statistical separation between the two classes is used to select the most suitable Landsat image
# a Random Forest machine learning algorithm is used to train the prediction model
# various derived reflectances and indices for the selected image are the candidate predictor variables   
# After training on the NCAS map for the same or a preceding year, the algorithm is applied to map forest cover.
# code translated from original Matlab code and then somewhat modified 

import argparse # to run command line
import os # to run command line
import numpy
import xarray
import datacube
import pandas
from time import time
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier #some info on http://blog.yhat.com/posts/random-forests-in-python.html
import collections
from datacube.storage import masking
from datacube.helpers import ga_pq_fuser

def print_progress(process,t):
    """generate some runtime diagnostics"""
    dt = (time()-t)    
    print(process+' completed in '+str(int(dt)) + ' seconds...')
    t = time()
    return t

def get_treemap(year,query):
    """get forest map for analysis area"""
    finpath = '/g/data/xc0/project/landcover/NCAS_woody_v1/national_composite_netcdf/'
    fin = finpath + 'NCAS_woody_veg_composite_{year}.nc'.format(year=year)
    try:
        NCAS_map = xarray.open_dataset(fin)
    except Exception:
        print('Could not find same year so using previous year as prior')
        fin = finpath + 'NCAS_woody_veg_composite_{year}.nc'.format(year=year-1)
        NCAS_map = xarray.open_dataset(fin)
    treemap = NCAS_map.sel(
        latitude=slice(query['y'][0],query['y'][1]),
        longitude=slice(query['x'][0],query['x'][1])
        )
    treemap = treemap.NCAS_woody.squeeze()
    return treemap
    
def getLandsatStack(landsat_number,query):
    """get Landsat band reflectances and pixel quality information"""
    dc = datacube.Datacube(app='TreeMapping.getLandsatStack')
    product= 'ls'+str(landsat_number)+'_nbart_albers'
    rquery = {**query, 
              'resampling' : 'bilinear',
              'measurements' : ['red','green','blue','nir','swir1','swir2']}
    stack = dc.load(product,group_by='solar_day',**rquery) # group by solar day: scenes for same day are merged - causes pixel quality issues
    stack['product'] = ('time', numpy.repeat(product, stack.time.size)) # adds a label identifying the product
    # now get pixel quality
    qquery = {**query,
              'resampling' : 'nearest',
              'measurements' : ['pixelquality']}
    product= 'ls'+str(landsat_number)+'_pq_albers'
    pq_stack = dc.load(product,group_by='solar_day',fuse_func=ga_pq_fuser,**qquery) # group by solar day: scenes for same day are merged - causes pixel quality issues
    # create land and good quality masks 
    # pandas.DataFrame.from_dict(masking.get_flags_def(pq_stack.pixelquality), orient='index') # to see the list of flags
    pq_stack['land']= masking.make_mask(pq_stack.pixelquality, land_sea='land')
    #pq_stack['ga_good_pixel']= masking.make_mask(pq_stack.pixelquality, ga_good_pixel=True) # not using this as it has issues
    clear_obs= masking.make_mask(pq_stack.pixelquality,cloud_acca='no_cloud')
    clear_obs= clear_obs*masking.make_mask(pq_stack.pixelquality,cloud_fmask='no_cloud')
    clear_obs= clear_obs*masking.make_mask(pq_stack.pixelquality,cloud_shadow_acca='no_cloud_shadow')
    clear_obs= clear_obs*masking.make_mask(pq_stack.pixelquality,cloud_shadow_fmask='no_cloud_shadow')
    pq_stack['no_cloud']=clear_obs
    # align the band and pixel quality stacks 
    # "join=inner" means that images without pixel quality information are rejected.
    lspq_stack, ls_stack = xarray.align(pq_stack,stack,join='inner') 
    lspq_stack['good_pixel']= lspq_stack.no_cloud.where(ls_stack.red>0,False,drop=False) # also remove negative reflectances (NaNs)
    return lspq_stack, ls_stack

def veg_indices(img):
    """Calculates several indices useful in tree cover classification"""
    DN2val = 1/1e4
    # NCAS-type indices
    img['Iw'] = 1 - (DN2val*img.red + DN2val*img.swir1)/2 
    img['Iw2'] = (2 + DN2val*img.red + DN2val*img.swir1 - 2*DN2val*img.nir)/2 
    # Tasselled cap indices (using coefficients of Baig et al, 2014)
    img['TCB'] = (0.3029*DN2val*img.blue + 0.2786*DN2val*img.green + 0.4733*DN2val*img.red 
                  + 0.508*DN2val*img.swir1 + 0.1872*DN2val*img.swir2) 
    img['TCG'] = (-0.2941*DN2val*img.blue + -0.243*DN2val*img.green + -0.5424*DN2val*img.red 
                  + 0.0713*DN2val*img.swir1 + -0.1608*DN2val*img.swir2) 
    img['TCW'] = (0.1511*DN2val*img.blue + 0.1973*DN2val*img.green + 0.3283*DN2val*img.red 
                  + -0.7117*DN2val*img.swir1 + -0.4559*DN2val*img.swir2)   
    img['TCA'] =  numpy.arctan(img.TCG/img.TCB)
    # burn indices
    img['NBR'] = (DN2val*img.nir-DN2val*img.swir2)/(DN2val*img.swir2+DN2val*img.nir) 
    img['BAI'] = 1/((0.1-DN2val*img.red)**2+(0.06-DN2val*img.nir)**2)     
    img['BAIM'] = 1/((0.05-DN2val*img.nir)**2+(0.2-DN2val*img.swir2)**2)     
    # vegetation indices
    img['NDVI'] = (DN2val*img.nir-DN2val*img.red)/(DN2val*img.nir+DN2val*img.red) 
    img['NDMI'] = (DN2val*img.nir-DN2val*img.swir1)/(DN2val*img.nir+DN2val*img.swir1)      
    img['EVI'] = 2.5*(DN2val*img.nir-DN2val*img.red)/(DN2val*img.nir+6*DN2val*img.red-7.5*DN2val*img.blue+1)
    img['GVMI'] = ((DN2val*img.nir+0.1)-(DN2val*img.swir1+0.02))/((DN2val*img.nir+0.1)+(DN2val*img.swir1+0.02))
    img['CMI'] = numpy.maximum(img.GVMI-(0.775*img.EVI-0.0757),0) 
    # Wentao Ye's forest index
    img['Wentao'] = numpy.maximum(((DN2val*img.nir - DN2val*img.red - 0.01)/( 
                    (DN2val*img.nir-DN2val*img.red)) * ((1-DN2val*img.nir)/(0.1+DN2val*img.green)) ),0)     
    return img

def update_prior(treemap,img,pq):
    """Removes supposed forest pixels that are unlikely to be forest pixels based on band indices"""
    NDVI_threshold = 0.15
    BAIM_threshold = 150
    EVI_threshold = 0.05
    prior = treemap.values
    prior = numpy.where(img.NDVI < NDVI_threshold,0,prior)
    prior = numpy.where(img.BAIM > BAIM_threshold,0,prior)
    prior = numpy.where(img.EVI < EVI_threshold,0,prior)
    prior = numpy.where(pq.good_pixel==1,prior,treemap.values)
    prior[numpy.isfinite(prior)==False]=0 # there appear to be a few stray nans in the forest cover maps
    return prior
#plt.imshow(img.NDVI) ; plt.axis('off'); plt.axis('equal'); plt.colorbar() # for debugging

def calc_SeparationScore(predictor,prior,pq):
    """Returns a measure of contrast between the forest and non-forest pixels for the image"""
    ForestValues=predictor.where((prior==1) & pq.good_pixel & pq.land,drop=True)
    NonforestValues=predictor.where((prior==0) & pq.good_pixel & pq.land, drop=True)
    AbsDiffMean=numpy.abs((ForestValues.mean().values-NonforestValues.mean().values))
    MeanStd = (ForestValues.std().values*NonforestValues.std().values)**0.5
    SeparationScore=(AbsDiffMean/MeanStd)
    return SeparationScore

def get_timeslice_score(pq,img,min_good_data_fraction,treemap):
    """Calculate separation scores for all images and throws out images with lots of missing data"""
    fraction_ok = (pq['good_pixel'] & pq['land']).values.mean() / (pq['land']).values.mean()
    BestOne = [0]
    if fraction_ok > min_good_data_fraction: # if more than threshold then analyse
        #print(fraction_ok)        # for debugging
        img = veg_indices(img)
        prior = update_prior(treemap,img,pq)
        for vi in img.variables:
            predictor = eval('img.' + vi)
            if predictor.size==prior.size:
                SeparationScore = calc_SeparationScore(predictor,prior,pq)
                if SeparationScore>BestOne[0]:
                    #print(SeparationScore)  # for debugging
                    BestOne=[SeparationScore,img.time.values,vi,fraction_ok,str(img.product.values)]
    return BestOne

def get_rfc_probabilities(prior,img,pq):
    """Does a Random forest classification and return probability of forest"""
    pix=numpy.where(pq.good_pixel & pq.land)
    predictand=prior[pix]
    predictors=[]
    for vi in img.variables:
        predictor = eval('img.' + vi)
        if predictor.size==prior.size:
            predictor_array=predictor.values[pix]
            predictor_array[numpy.isfinite(predictor_array)==False]=0 # set inf and nan to zero (probably not ideal to assign like this, but hopefully ok)
            predictors.append(predictor_array)
    predictors=numpy.transpose(predictors)
    predictors = pandas.DataFrame(predictors)
    clf = RandomForestClassifier(n_jobs=-1,max_depth=3) # -1 means use all cores available. Rest is all default settings
    clf.fit(predictors,predictand)
    predicted=clf.predict_proba(predictors)
    tree_prob=prior.astype(float)
    tree_prob[pix]=predicted[:,1]
    return tree_prob

def map_tile(ul_lon,ul_lat,tilesize,year):
    """Maps a tile within the full domain of interest."""
    t0 = time()     # Start clock for diagnostics
    t=t0 
    # general spatial and temporal query parameters. 
    query = { 'x' : (ul_lon, ul_lon+tilesize), 
              'y' : (ul_lat, ul_lat-tilesize),
              # Starts in Jul to allow for first half year bushfires so that result is more representative for the entire year.
              'time' : (str(year)+'-07', str(year)+'-12'),  
              'output_crs' : 'WGS84',
              'resolution' : (-0.00025, 0.00025)}
    process='Loading NCAS treemap '
    treemap=get_treemap(year,query)
    t = print_progress(process,t)
    #treemap.plot.imshow() # visual check
    if (treemap==1).sum()==0:
        print('no trees - return prior')
        treemap_new = treemap.copy() 
    else:
        process='Loading landsat imagery and pixel quality information'
        landsat_numbers=[7,8] # get both landsat 7 and 8 (but later on, preference 8)
        pq_stack = []
        stack = []
        for landsat_number in landsat_numbers:
            lspq_stack, ls_stack = getLandsatStack(landsat_number,query)
            pq_stack.append(lspq_stack)   
            stack.append(ls_stack)   
        pq_stack = xarray.concat(pq_stack, dim='time').sortby('time')
        stack = xarray.concat(stack, dim='time').sortby('time')
        landmask=pq_stack.land.max(dim='time').values
        pq_stack=pq_stack.drop('land')
        t = print_progress(process,t)
        # adjust treemap to fit the imagery
        reindex_tolerance=numpy.abs(query['resolution'][0])/2
        treemap=treemap.reindex_like(stack,method='nearest',tolerance=reindex_tolerance)
        process='Calculating suitability scores for all predictors and all images'
        min_good_data_fraction=0.6 # determine show many of the land pixels must have good quality to use this image
        BestOnes=[] # initialise list 
        NumberImages=len(stack.time.values)
        for ti in range(0,NumberImages):
            pq=pq_stack.isel(time=ti)
            pq['land']=landmask+0*pq.pixelquality # add landmask
            img=stack.isel(time=ti)
            BestForImage = get_timeslice_score(pq,img,min_good_data_fraction,treemap) 
            if len(BestForImage)>1:
                BestOnes.append(BestForImage) # to do (maybe): combine this with above call to use xarray.apply
        BestOnes=pandas.DataFrame(BestOnes,columns=['score', 'time', 'predictor', 'image_quality', 'product'])
        # give preference to Landsat 8 and dates with good contrast
        BestOnes.sort_values(by=['product','score'],ascending=[False,False], inplace=True) 
        t = print_progress(process,t)
        #BestForImage # for debugging/testing
        #BestOnes # for debugging/testing
        BestFew=5 # How many images you want to classify
        BestFew=numpy.min((BestFew,len(BestOnes))) # use than preferred number not enough images lacking.
        process='Consensus forest probability mapping based on best '+str(BestFew)+' images'
        # initialise variables
        cum_tree_prob=0*treemap.values.astype(float)
        Nobs=0*treemap.values.astype(float)
        # perform classification
        for rankval in range(0,BestFew):
            img = stack.sel(time = BestOnes.time.values[rankval])
            pq = pq_stack.sel(time = BestOnes.time.values[rankval])
            pq['land']=landmask+0*pq.pixelquality
            img = veg_indices(img)
            predictor = eval('img.' + BestOnes.predictor.values[rankval])
            prior = update_prior(treemap,img,pq)
            tree_prob = get_rfc_probabilities(prior,img,pq)
            pqok=numpy.where(pq.good_pixel)
            Nobs[pqok]=Nobs[pqok]+1
            cum_tree_prob[pqok]=cum_tree_prob[pqok]+tree_prob[pqok]
            #print('.') 
        tree_prob=cum_tree_prob/Nobs # consensus mean probability
        ThresholdProbability=0.7 # Determines how much probability there needs to be to flip to other class
        treemap_new = treemap.copy() # to avoid changing attributes of original
        ChangeEvidenceTooWeak=(abs(tree_prob-treemap)<ThresholdProbability) | numpy.isnan(tree_prob)  # these cells won't be changed
        AlternateValue=-treemap+1 # the alternate value (0>1,1>0)
        treemap_new=treemap_new.where(ChangeEvidenceTooWeak,AlternateValue)
        treemap_new=treemap_new.where(pq.land.values) # remove sea pixels
        treemap_new.coords['time']= pandas.to_datetime(str(year)+'-01-01')
        # treemap_new.plot.imshow() # for testing
        t = print_progress(process,t)
    #Run time summary
    t = (time()-t0)
    t_per_degree=t/((tilesize**2)*60) # would be minutes per 1x1 degree area 
    print('Tile completed in '+str(int(t)) + ' seconds.')
    return treemap_new

def main(domain_ul_lat,domain_ul_lon,domain_size):
    """Callable workflow that maps over the domain specified (domain size and year are hard coded below)."""
    #domain_ul_lat=-32 # for testing
    #domain_ul_lon=149.5 # for testing
    # general settings
    tilesize = 0.25 # size of the tiles processed (must be a integer division of domain_size)
    year = 2017
    # general spatial and temporal query parameters. 
    domain_query = { 'x' : (domain_ul_lon, domain_ul_lon+domain_size), 
              'y' : (domain_ul_lat, domain_ul_lat-domain_size),
              'output_crs' : 'WGS84',
              'resolution' : (-0.00025, 0.00025)}
    if (domain_size/tilesize).is_integer()==False: # check sensible values are chosen.
        raise ValueError('Domain size must be divisible by tile size')    
    # Start clock
    t0 = time()
    t=t0 
    # Classify new map        
    domain_treemap=get_treemap(year,domain_query)
    output_map = domain_treemap.copy() 
    for ul_lat in numpy.arange(domain_ul_lat,domain_ul_lat-domain_size,-tilesize):
        for ul_lon in numpy.arange(domain_ul_lon,domain_ul_lon+domain_size,tilesize):
            print('Doing tile with ul corner ['+str(ul_lat)+','+str(ul_lon)+']')   
            tile_map = map_tile(ul_lon,ul_lat,tilesize,year)
            if tile_map.size>0:
                reindex_tolerance=numpy.abs(domain_query['resolution'][0])/2
                tile_map =tile_map.reindex_like(domain_treemap,method='nearest',tolerance=reindex_tolerance)
                output_map = tile_map.combine_first(output_map)
    # Save map        
    outpath='/g/data/xc0/project/GA_burn/temp/'
    fn='treemap_'+str(year)+'_'+str(domain_ul_lat)+'S_'+str(domain_ul_lon)+'E_'+str(domain_size)+'deg.nc'
    FillValue =-1
    process='Saving as netcdf '
    output_map.to_netcdf(outpath+fn,encoding={'NCAS_woody': {'dtype': 'int','_FillValue':FillValue,'zlib': True,'complevel': 1}})
    t = print_progress(process+outpath+fn,t)
    #Runtime summary
    t = (time()-t0)
    t_per_degree=t/((domain_size**2)*60) # would be minutes per 1x1 degree area 
    print('===========================')
    print('Full domain completed in '+str(int(t)) + ' seconds.')
    print('That is equivalent to ' + str(int(t_per_degree)) +' minutes per square degree.')
    print('===========================')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run TreeMapping command line')
    parser.add_argument(
        '--domain_ul_lat', type=float,
        default=os.environ.get('DOMAIN_UL_LAT'),
        help='latitude of the top left (i.e. northwest) corner of the processing domain (deg, S is negative)',
        metavar='<domain_ul_lat>')
    parser.add_argument(
        '--domain_ul_lon', type=float,
        default=os.environ.get('DOMAIN_UL_LON'),
        help='longitude of the top left (i.e. northwest) corner of the processing domain (deg, E is positive)',
        metavar='<domain_ul_lon>')
    parser.add_argument(
        '--domain_size', type=float,
        default=os.environ.get('DOMAIN_SIZE'),
        help='size of the (square) mapping area in degrees',
        metavar='<domain_size>')
    args = parser.parse_args()
    print(args)    
    if args.domain_ul_lat is None:
        raise ValueError('domain_ul_lat must be provided as an argument or set as the environment variable $DOMAIN_UL_LAT.')    
    if args.domain_ul_lon is None:
        raise ValueError('domain_ul_lon must be provided as an argument or set as the environment variable $DOMAIN_UL_LON.')    
    if args.domain_size is None:
        raise ValueError('domain_size must be provided as an argument or set as the environment variable $DOMAIN_SIZE.')    
    main(args.domain_ul_lat,args.domain_ul_lon,args.domain_size)    
    
    
    
    


# In[346]:


# NOTES    
# Just some example areas for testing. Most of them had some type of forest loss.
# NW Tasmania
#ul_lat = -41.2
#ul_lon = 145.0
# Uarbry (Sir Ivan fire, Feb 2017)
#ul_lat = -32.0
#ul_lon = 149.6
# near Uarbry (unaffected)
#ul_lat = -31.7
#ul_lon = 148.8
# ACT
#ul_lat = -35.2
#ul_lon = 149.1
#Shoalhaven Qld (Byfield fire, Aug 2017)
#ul_lat = -22.7
#ul_lon = 150.7
# Van Nuyts/Madura (WA)
#ul_lat = -32.1
#ul_lon = 126.7

