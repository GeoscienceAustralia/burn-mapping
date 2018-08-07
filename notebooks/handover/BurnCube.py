        return nbart_stack
        
        
    def load_cube(self, x, y, res, period, n_landsat):
        
        nbart_stack = self._load_nbart(x, y, res, period, n_landsat)
        pq_stack = self._load_pq(x, y, res, period, n_landsat)
        
        pq_stack, nbart_stack = xr.align(pq_stack, nbart_stack, join='inner') 
        pq_stack['good_pixel'] = pq_stack.no_cloud.where(nbart_stack.red > 0, False, drop=False)
        
        goodpix = pq_stack.no_cloud * (pq_stack.pixelquality > 0) * pq_stack.good_pixel
        
     
        mask = np.nanmean(goodpix.values.reshape(goodpix.shape[0], -1), axis=1) > .2


        cubes = [nbart_stack[band][mask, :, :]*goodpix[mask, :, :] for band in ['red','green','blue','nir','swir1','swir2']]
        X = np.stack(cubes, axis=0)

        
        #data = xr.Dataset(coords={'band': ['red','green','blue','nir','swir1','swir2'],
        data = xr.Dataset(coords={'band': np.arange(6),
                                  'time':nbart_stack.time[mask],
                                  'y':nbart_stack.y[:],
                                  'x':nbart_stack.x[:]},
                          attrs={'crs':'EPSG:3577'})
        data["cube"] = (('band','time','y','x'),X)
        data.time.attrs=[]
        
        self.dataset = data

    def geomedian(self, period, n_procs=4, epsilon=.5, max_iter=40):
        # Define an output queue
       
        n = len(self.dataset.y)*len(self.dataset.x)
        out_arr = mp.Array(ctypes.c_float, len(self.dataset.band)*n)
        gmed = np.frombuffer(out_arr.get_obj(), dtype=np.float32).reshape((len(self.dataset.band),n))
        gmed.fill(np.nan)
    
        _X = self.dataset['cube'].sel(time=slice(period[0],period[1]))
        #print(_X.shape)
        t_dim = _X.time[:]
        in_arr = mp.Array(ctypes.c_short, len(self.dataset.band)*len(_X.time)*n)
        X = np.frombuffer(in_arr.get_obj(), dtype=np.int16).reshape(len(self.dataset.band), len(_X.time), n)
        X[:] = _X.data.reshape(len(self.dataset.band), len(_X.time), -1)

        def init(shared_in_arr_, shared_out_arr_):
            global shared_in_arr
            global shared_out_arr
            shared_in_arr = shared_in_arr_ # must be inherited, not passed as an argument
            shared_out_arr = shared_out_arr_ # must be inherited, not passed as an argument


        # write to arr from different processes
        with closing(mp.Pool(initializer=init, initargs=(in_arr, out_arr,))) as p:
        # many processes access different slices of the same array
            chunk = n//n_procs
            #p.map_async(g, [(i, i + step) for i in range(0, N, step)])
            p.map_async(dist_geomedian, [(i, min(n, i+chunk), X.shape) for i in range(0, n, chunk)])
        p.join()
        
        ds = xr.Dataset(coords={'time':t_dim,'y':self.dataset.y[:],'x':self.dataset.x[:],'bands':self.dataset.band}, attrs={'crs':'EPSG:3577'})
        ds['geomedian'] = (('bands','y','x'),gmed[:].reshape((len(self.dataset.band),len(self.dataset.y),len(self.dataset.x))).astype(np.float32))
        
        del gmed, in_arr, out_arr, X
        self.geomed = ds
        
    def distances(self, period, n_procs=4):
        n = len(self.dataset.y)*len(self.dataset.x)
        _X = self.dataset['cube'].sel(time=slice(period[0],period[1]))
        
        t_dim = _X.time.data
        nir = _X[3,:,:,:].data.astype('float32')
        swir2 = _X[5,:,:,:].data.astype('float32')
        nir[nir<=0]=np.nan
        swir2[swir2<=0]=np.nan
        nbr = ((nir-swir2) / (nir+swir2))
        print(nbr.shape)
        
        out_arr1 = mp.Array(ctypes.c_float, len(t_dim)*n)
        out_arr2 = mp.Array(ctypes.c_float, len(t_dim)*n)
        out_arr3 = mp.Array(ctypes.c_short, len(t_dim)*n)
        
        cos_dist = np.frombuffer(out_arr1.get_obj(), dtype=np.float32).reshape((len(t_dim),n))
        cos_dist.fill(np.nan)
        nbr_dist = np.frombuffer(out_arr2.get_obj(), dtype=np.float32).reshape((len(t_dim),n))
        nbr_dist.fill(np.nan)
        direction = np.frombuffer(out_arr3.get_obj(), dtype=np.int16).reshape((len(t_dim),n))
        direction.fill(0)
        
        in_arr1 = mp.Array(ctypes.c_short, len(self.dataset.band)*len(_X.time)*n)
        X = np.frombuffer(in_arr1.get_obj(), dtype=np.int16).reshape(len(self.dataset.band), len(_X.time), n)
        X[:] = _X.data.reshape(len(self.dataset.band), len(_X.time), -1)
        
        in_arr2 = mp.Array(ctypes.c_float, len(self.dataset.band)*n)        
        gmed = np.frombuffer(in_arr2.get_obj(), dtype=np.float32).reshape(len(self.dataset.band), n)
        gmed[:] = self.geomed['geomedian'].data.reshape(len(self.dataset.band),  -1)
        
        
        def init(shared_in_arr1_, shared_in_arr2_, shared_out_arr1_, shared_out_arr2_, shared_out_arr3_):
            global shared_in_arr1
            global shared_in_arr2
            global shared_out_arr1
            global shared_out_arr2
            global shared_out_arr3
            
            shared_in_arr1 = shared_in_arr1_ # must be inherited, not passed as an argument            
            shared_in_arr2 = shared_in_arr2_ # must be inherited, not passed as an argument
            shared_out_arr1 = shared_out_arr1_ # must be inherited, not passed as an argument
            shared_out_arr2 = shared_out_arr2_ # must be inherited, not passed as an argument           
            shared_out_arr3 = shared_out_arr3_ # must be inherited, not passed as an argument
            
         # write to arr from different processes
        with closing(mp.Pool(initializer=init, initargs=(in_arr1, in_arr2, out_arr1, out_arr2, out_arr3,))) as p:
        # many processes access different slices of the same array
            chunk = n//n_procs
      
            p.map_async(dist_distance, [(i, min(n, i+chunk), X.shape) for i in range(0, n, chunk)])
            
        p.join()
       
        
        
        ds = xr.Dataset(coords={'time':t_dim,'y':self.dataset.y[:],'x':self.dataset.x[:],'bands':self.dataset.band}, attrs={'crs':'EPSG:3577'})
        ds['cosdist'] = (('time','y','x'),cos_dist[:].reshape((len(t_dim),len(self.dataset.y),len(self.dataset.x))).astype('float32'))
        ds['NBRDist'] = (('time','y','x'),nbr_dist[:].reshape((len(t_dim),len(self.dataset.y),len(self.dataset.x))).astype('float32'))
        ds['ChangeDir'] = (('time','y','x'),direction[:].reshape((len(t_dim),len(self.dataset.y),len(self.dataset.x))).astype('float32'))
        ds['NBR'] = (('time','y','x'),nbr)
        #print(ds)
        del in_arr1,in_arr2,out_arr1,out_arr2,out_arr3,gmed,X,cos_dist,nbr_dist,direction,nbr
        self.dists = ds

    def outliers(self):
        NBRoutlier = np.nanpercentile(self.dists.NBRDist,75,axis=0)+1.5*(np.nanpercentile(self.dists.NBRDist,75,axis=0)-np.nanpercentile(self.dists.NBRDist,25,axis=0))
        
        CDistoutlier = np.nanpercentile(self.dists.cosdist,75,axis=0)+1.5*(np.nanpercentile(self.dists.cosdist,75,axis=0)-np.nanpercentile(self.dists.cosdist,25,axis=0))
        
        ds = xr.Dataset(coords={'y':self.dataset.y[:],'x':self.dataset.x[:]}, attrs={'crs':'EPSG:3577'})
        ds['CDistoutlier'] = (('y','x'),CDistoutlier.astype('float32'))
        ds['NBRoutlier'] = (('y','x'),NBRoutlier.astype('float32'))
        self.outlrs = ds

    def region_growing(self,severity):
    
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
        ti = np.where(self.dists.time>np.datetime64(d))[0][0]
        NBR_score=(self.dists.ChangeDir*self.dists.NBRDist)[ti,:,:]/self.outlrs.NBRoutlier
        cos_score=(self.dists.ChangeDir*self.dists.cosdist)[ti,:,:]/self.outlrs.CDistoutlier
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
            ti = np.where(self.dists.time>np.datetime64(d))[0][0]
            NBR_score=(self.dists.ChangeDir*self.dists.NBRDist)[ti,:,:]/self.outlrs.NBRoutlier
            cos_score=(self.dists.ChangeDir*self.dists.cosdist)[ti,:,:]/self.outlrs.CDistoutlier
            Potential=((NBR_score>z_distance)&(cos_score>z_distance)).astype(int)
            all_labels = measure.label(Potential.astype(int).values,background=0)
            NewPotential=0.*SeedMap.astype(float)
            for ri in range(1,np.max(np.unique(all_labels))): 
                NewPotential[all_labels==ri]=np.mean(np.extract(all_labels==ri,SeedMap))
            AnnualMap=AnnualMap+(NewPotential>fraction_seedmap).astype(int)
        BurnExtent=(AnnualMap>0).astype(int)
        #BurnArea = BurnExtent*SeedMap+BurnExtent
        
        return BurnExtent
    
    def severitymapping(self,period, n_procs=4,method='NBR',growing=True):
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
        data = self.dists.sel(time=slice(period[0],period[1]))               
        CDist = self.dists.cosdist.data.reshape((len(data.time),-1))
        CDistoutlier = self.outlrs.CDistoutlier.data.reshape((len(data.x)*len(data.y)))
        NBRDist = self.dists.NBRDist.data.reshape((len(data.time),-1))
        NBR = self.dists.NBR.data.reshape((len(data.time),-1))
        NBRoutlier = self.outlrs.NBRoutlier.data.reshape((len(data.x)*len(data.y)))
        ChangeDir = self.dists.ChangeDir.data.reshape((len(data.time),-1))
        

        if method=='NBR':
            tmp =self.dists.cosdist.where((self.dists.cosdist>self.outlrs.CDistoutlier)&(self.dists.NBR<0)).sum(axis=0).data            
            tmp = tmp.reshape((len(self.dataset.x)*len(self.dataset.y)))
            outlierind=np.where(tmp>0)[0]
            CDist=CDist[:,outlierind]
            CDistoutlier=CDistoutlier[outlierind]
            NBR=NBR[:,outlierind]
            NBRDist=NBRDist[:,outlierind]
            NBRoutlier= NBRoutlier[outlierind]
            ds = xr.Dataset(coords={'time':self.dists.time[:],'points':np.linspace(0,len(outlierind)-1,len(outlierind)).astype('uint32')})    
            ds['NBR'] = (('time','points'),NBR)        
            ds['CosDist'] = (('time','points'),CDist)
            ds['CDistoutlier'] = (('points'),CDistoutlier)        
        elif method =='NBRdist':
            #print(CDist.shape,CDistoutlier.shape)
            tmp =self.dists.cosdist.where((self.dists.cosdist>self.outlrs.CDistoutlier)&(self.dists.NBRDist>self.outlrs.NBRoutlier)&(self.dists.ChangeDir==1)).sum(axis=0).data            
            tmp = tmp.reshape((len(self.dataset.x)*len(self.dataset.y)))
            outlierind=np.where(tmp>0)[0]  
            CDist=CDist[:,outlierind]
            CDistoutlier=CDistoutlier[outlierind]
            NBR=NBR[:,outlierind]
            NBRDist = NBRDist[:,outlierind]
            NBRoutlier = NBRoutlier[outlierind]
            ChangeDir = ChangeDir[:,outlierind] 
            ds = xr.Dataset(coords={'time':data.time[:],'points':np.linspace(0,len(outlierind)-1,len(outlierind)).astype('uint32')})    
            ds['NBRDist'] = (('time','points'),NBRDist)
            ds['CosDist'] = (('time','points'),CDist)
            ds['ChangeDir'] = (('time','points'),ChangeDir)
            ds['CDistoutlier'] = (('points'),CDistoutlier)
            ds['NBRoutlier'] = (('points'),NBRoutlier)
        else:
            raise ValueError 
        if len(outlierind)>0:
            sev = np.zeros((len(outlierind)))
            dates = np.zeros((len(outlierind)))
            days = np.zeros((len(outlierind)))

            for i in range(0,len(outlierind)):            
                #print(i)             
                sev[i], dates[i], days[i]=severity(data=ds.sel(points=i),method=method)


            sevindex=np.zeros((len(self.dists.x)*len(self.dists.y)))
            duration=np.zeros((len(self.dists.x)*len(self.dists.y)))
            startdate=np.zeros((len(self.dists.x)*len(self.dists.y)))

            sevindex[outlierind]=sev
            duration[outlierind]=days
            startdate[outlierind]=dates

            sevindex=sevindex.reshape((len(self.dists.y),len(self.dists.x)))
            duration=duration.reshape((len(self.dists.y),len(self.dists.x)))
            startdate=startdate.reshape((len(self.dists.y),len(self.dists.x)))
            #startdate[startdate==0]=np.nan
            out = xr.Dataset(coords={'y':self.dists.y[:],'x':self.dists.x[:]},attrs={'crs':'EPSG:3577'})

            out['StartDate']=(('y','x'),startdate)
            out['Duration']=(('y','x'),duration.astype('uint16'))
            burnt = np.zeros((len(data.y),len(data.x)))
            burnt[duration>1] = 1
            out['Severity']=(('y','x'),sevindex.astype('float32'))
            out['Severe']=(('y','x'),burnt.astype('uint16'))
            if growing==True:
                BurnArea = self.region_growing(out)
                out['Medium'] = (('y','x'),BurnArea.astype('int16')) 
            #Add hotspot as collaborated burnt area
            extent = [np.min(self.dists.x.data),np.max(self.dists.x.data),np.min(self.dists.y.data),np.max(self.dists.y.data)]
            year = int(period[0][0:4])
            polygons = hotspot_polygon(year,extent,4000) # generate hotspot polygons with 4km buffer
            #generate the mask for hotspot data for filtering or overlay
            if polygons.is_empty:
                print('No hotspots data.')
            else:
                coords = out.coords

                if polygons.type=='MultiPolygon':
                    HotspotMask=np.zeros((len(self.dists.y),len(self.dists.x)))
                    for polygon in polygons:
                        HotspotMask_tmp =  outline_to_mask(polygon.exterior, coords['x'], coords['y'])
                        HotspotMask = HotspotMask_tmp + HotspotMask
                    HotspotMask=xr.DataArray(HotspotMask, coords=coords, dims=('y', 'x'))
                if polygons.type=='Polygon':
                    HotspotMask =  outline_to_mask(polygons.exterior, coords['x'], coords['y'])
                    HotspotMask=xr.DataArray(HotspotMask, coords=coords, dims=('y', 'x'))
                out['Collaborate'] = (('y','x'),HotspotMask.astype('uint8'))
            
        else:
            print('no burnt area detected')
            out = np.array([]) # no detected burn
            
        return out

        
