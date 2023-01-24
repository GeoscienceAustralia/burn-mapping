import ctypes
import datetime
import logging
import multiprocessing as mp
from contextlib import closing

import numpy as np
import pandas as pd
import pyproj
import s3fs
import xarray as xr
from shapely import geometry
from shapely.geometry import Point
from shapely.ops import unary_union

logger = logging.getLogger(__name__)


def task_to_ranges(task_id):

    _ = s3fs.S3FileSystem(anon=True)

    task_map = pd.read_csv(
        "s3://dea-public-data-dev/projects/burn_cube/configs/10-year-historical-processing.csv",
        parse_dates=[
            "Period Start",
            "Period End",
            "Mapping Period Start",
            "Mapping Period End",
        ],
        dayfirst=True,
    )

    result_dict = {}

    if task_id not in list(task_map["Processing Name"]):
        return result_dict

    task_info = task_map[task_map["Processing Name"] == task_id].iloc[0]

    upstream_products = ["GeoMED", "WOfS summary"]

    for upstream_product in upstream_products:
        result_dict[upstream_product] = task_info[upstream_product]

    periods_columns = [
        "Period Start",
        "Period End",
        "Mapping Period Start",
        "Mapping Period End",
    ]

    for periods_column in periods_columns:
        result_dict[periods_column] = task_info[periods_column].strftime("%Y-%m-%d")

    return result_dict


def dynamic_task_to_ranges(dtime):
    import calendar

    period_last_day = calendar.monthrange(dtime.year - 1, dtime.month)[1]

    result_dict = {}

    result_dict["Period Start"] = datetime.datetime(
        dtime.year - 5, dtime.month + 1, 1
    ).strftime("%Y-%m-%d")
    result_dict["Period End"] = datetime.datetime(
        dtime.year - 1, dtime.month, period_last_day
    ).strftime("%Y-%m-%d")

    mapping_period_last_day = calendar.monthrange(dtime.year, dtime.month)[1]
    result_dict["Mapping Period Start"] = datetime.datetime(
        dtime.year - 1, dtime.month + 1, 1
    ).strftime("%Y-%m-%d")
    result_dict["Mapping Period End"] = datetime.datetime(
        dtime.year, dtime.month, mapping_period_last_day
    ).strftime("%Y-%m-%d")

    if dtime.month < 5:
        result_dict["GeoMED"] = f"ga_ls8c_gm_cyear_3_{dtime.year - 5}--P4Y"
        result_dict["WOfS summary"] = f"ga_ls_wo_fq_cyear_3_{dtime.year - 1}--P1Y"
    elif dtime.month == 12:
        result_dict["GeoMED"] = f"ga_ls8c_gm_cyear_3_{dtime.year - 4}--P4Y"
        result_dict["WOfS summary"] = f"ga_ls_wo_fq_cyear_3_{dtime.year}--P1Y"
    else:
        result_dict["GeoMED"] = f"ga_ls8c_gm_fyear_3_{dtime.year - 5}-07--P4Y"
        result_dict["WOfS summary"] = f"ga_ls_wo_fq_fyear_3_{dtime.year}-07--P1Y"

    return result_dict


def generate_task(task_id):
    updated_task_id = f"{task_id.split('-')[0]}-20{task_id.split('-')[1]}"
    dtime = datetime.datetime.strptime(updated_task_id, "%b-%Y")

    if dtime.year > 2023:
        result_dict = dynamic_task_to_ranges(dtime)
    else:
        result_dict = task_to_ranges(task_id)
    return result_dict


def stretch_rgb(data):
    """
    stretch RGB to 0-1
    """
    tmp = data
    a = (
        (tmp - np.nanpercentile(tmp, 1))
        / (np.nanpercentile(tmp, 99.5) - np.nanpercentile(tmp, 0.5))
        * 255
    )
    a[a > 255] = 255
    a[a < 0] = 0
    return a


def rgb_plot(r, g, b):
    from PIL import Image

    rgb_array = np.zeros((r.shape[0], r.shape[1], 3), "uint8")
    rgb_array[..., 0] = stretch_rgb(r)
    rgb_array[..., 1] = stretch_rgb(g)
    rgb_array[..., 2] = stretch_rgb(b)
    img = Image.fromarray(np.flipud(rgb_array))
    return img.resize((500, 500))


def cos_distance(ref, obs):
    """
    Returns the cosine distance between observation and reference
    The calculation is point based, easily adaptable to any dimension.
    Args:
        ref: reference (1-D array with multiple bands) e.g., geomatrix median [Nbands]
        obs: observation (with multiple bands, e.g. 6) e.g.,  monthly geomatrix median or reflectance [Nbands,ndays]

    Returns:
        cosdist: the cosine distance at each time step in [ndays]
    """
    ref = ref.astype(np.float32)[:, np.newaxis]
    obs = obs.astype(np.float32)
    cosdist = np.empty((obs.shape[1],))
    cosdist.fill(np.nan)

    cosdist = np.transpose(
        1
        - np.nansum(ref * obs, axis=0)
        / np.sqrt(np.sum(ref ** 2))
        / np.sqrt(np.nansum(obs ** 2, axis=0))
    )
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
    direction = np.zeros((obs.shape[0],), dtype="uint8")
    nbr_dist.fill(np.nan)
    index = np.where(~np.isnan(obs))[0]
    euc_dist = obs[index] - ref
    euc_norm = np.sqrt(euc_dist ** 2)
    nbr_dist[index] = euc_norm
    direction[index[euc_dist < -0.05]] = 1

    return nbr_dist, direction


def severity(
    nbr,
    nbr_dist,
    cos_dist,
    change_dir,
    nbr_outlier,
    cos_dist_outlier,
    t,
    method="NBRdist",
):
    """
    Returns the severity,duration and start date of the change.
    Args:
        nbr: normalised burn ratio in tx1 dimension
        nbr_dist: nbr distance in tx1 dimension
        cos_dist: cosine distance in tx1 dimension
        change_dir: NBR change direction in tx1 dimension
        nbr_outlier: outlier values for NBRdist
        cos_dist_outlier: outlier values for CDist
        t: dates of observations
        data: xarray including the cosine distances, NBR distances, NBR, change direction and outliers value
        method: two options to choose
            NBR: use cosine distance together with NBR<0
            NBRdist: use both cosine distance, NBR euclidean distance, and NBR change direction for change detection

    Returns:
        sevindex: severity
        startdate: first date change was detected
        duration: duration between the first and last date the change exceeded the outlier threshold
    """

    sevindex = 0
    startdate = 0
    duration = 0

    notnanind = np.where(~np.isnan(cos_dist))[0]  # remove the nan values for each pixel

    if method == "NBR":  # cosdist above the line and NBR<0
        outlierind = np.where(
            (cos_dist[notnanind] > cos_dist_outlier) & (nbr[notnanind] < 0)
        )[0]
        cosdist = cos_dist[notnanind]

    elif (
        method == "NBRdist"
    ):  # both cosdist and NBR dist above the line and it is negative change
        outlierind = np.where(
            (cos_dist[notnanind] > cos_dist_outlier)
            & (nbr_dist[notnanind] > nbr_outlier)
            & (change_dir[notnanind] == 1)
        )[0]

        cosdist = cos_dist[notnanind]
    else:
        raise ValueError
    t = t.astype("datetime64[ns]")
    t = t[notnanind]
    outlierdates = t[outlierind]
    n_out = len(outlierind)
    area_above_d0 = 0
    if n_out >= 2:
        tt = []
        for ii in range(0, n_out):
            if outlierind[ii] + 1 < len(t):
                u = np.where(t[outlierind[ii] + 1] == outlierdates)[
                    0
                ]  # next day have to be outlier to be included
                # print(u)

                if len(u) > 0:
                    t1_t0 = (
                        (t[outlierind[ii] + 1] - t[outlierind[ii]])
                        / np.timedelta64(1, "s")
                        / (60 * 60 * 24)
                    )
                    y1_y0 = (
                        cosdist[outlierind[ii] + 1] + cosdist[outlierind[ii]]
                    ) - 2 * cos_dist_outlier
                    area_above_d0 = (
                        area_above_d0 + 0.5 * y1_y0 * t1_t0
                    )  # calculate the area under the curve
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
    x_val, y_val = np.meshgrid(x, y)
    points = np.array((x_val.flatten(), y_val.flatten())).T
    mask = mpath.contains_points(points).reshape(x_val.shape)

    return mask


def nanpercentile(inarr, q):
    """
    faster nanpercentile than np.nanpercentile for axis 0 of a 3D array.
    modified from https://krstn.eu/np.nanpercentile()-there-has-to-be-a-faster-way/
    """
    arr = inarr.copy()
    # valid (non NaN) observations along the first axis
    valid_obs = np.isfinite(arr).sum(axis=0)
    # replace NaN with maximum
    max_val = np.nanmax(arr)
    arr[np.isnan(arr)] = max_val
    # sort - former NaNs will move to the end
    arr.sort(axis=0)

    # loop over requested quantiles
    if type(q) is list:
        qs = q
    else:
        qs = [q]
    quant_arrs = np.empty(shape=(len(qs), arr.shape[1], arr.shape[2]))
    quant_arrs.fill(np.nan)

    for i in range(len(qs)):
        quant = qs[i]
        # desired position as well as floor and ceiling of it
        k_arr = (valid_obs - 1) * (quant / 100.0)
        f_arr = np.floor(k_arr).astype(np.int32)
        c_arr = np.ceil(k_arr).astype(np.int32)
        fc_equal_k_mask = f_arr == c_arr

        # linear interpolation (like numpy percentile) takes the fractional part of desired position
        floor_val = _zvalue_from_index(arr, f_arr) * (c_arr - k_arr)
        ceil_val = _zvalue_from_index(arr, c_arr) * (k_arr - f_arr)

        quant_arr = floor_val + ceil_val
        quant_arr[fc_equal_k_mask] = _zvalue_from_index(arr, f_arr)[fc_equal_k_mask]

        quant_arrs[i] = quant_arr

    if quant_arrs.shape[0] == 1:
        return np.squeeze(quant_arrs, axis=0)
    else:
        return quant_arrs


def _zvalue_from_index(arr, ind):
    """
    private helper function to work around the limitation of np.choose() by employing np.take()
    arr has to be a 3D array
    ind has to be a 2D array containing values for z-indicies to take from arr
    modified from https://krstn.eu/np.nanpercentile()-there-has-to-be-a-faster-way/
    with order of nR and nC fixed.
    """
    # get number of columns and rows
    _, nr, nc = arr.shape

    # get linear indices and extract elements with np.take()
    idx = nr * nc * ind + nc * np.arange(nr)[:, np.newaxis] + np.arange(nc)
    return np.take(arr, idx)


def post_filtering(sev, hotspots_filtering=True, date_filtering=True):
    """
    This function cleans up the potential cloud contaminated results with hotspots data and start date
    variables:
        sev: outputs from BurnCube
        hotspots_filtering: whether filtering the results with hotspots data
        date_filtering: whether filtering the results with only five major changes with startdate info
    outputs:
        sev: with one extra layer 'cleaned'
    """
    if "Moderate" in sev.keys():
        burn_pixel = (
            sev.Moderate
        )  # burnpixel_masking(sev,'Moderate') # mask the burnt area with "Medium" burnt area
        filtered_burnscar = np.zeros(burn_pixel.data.shape).astype("f4")

        if hotspots_filtering:
            from skimage import measure

            all_labels = measure.label(burn_pixel.data, background=0)

            if ("Corroborate" in sev.keys()) * (sev.Corroborate.data.sum() > 0):
                hs_pixel = sev.Corroborate  # burnpixel_masking(sev,'Corroborate')
                tmp = all_labels * hs_pixel.data.astype("int32")
                overlappix = (-hs_pixel.data + burn_pixel.data * 2).reshape(-1)
                if len(overlappix[overlappix == 2]) > 0:
                    overlaplabels = np.unique(tmp)
                    labels = overlaplabels[overlaplabels > 0]
                    for i in labels:
                        seg = np.zeros(burn_pixel.data.shape)
                        seg[all_labels == i] = 1
                        if np.sum(seg * hs_pixel.data) > 0:
                            filtered_burnscar[seg == 1] = 1
                else:
                    filtered_burnscar[:] = burn_pixel.data.copy()

            else:
                filtered_burnscar = np.zeros(burn_pixel.data.shape)

            cleaned = np.zeros(burn_pixel.data.shape)
            filtered_burnscar[filtered_burnscar == 0] = np.nan
            clean_date = filtered_burnscar * sev.StartDate
            mask = np.where(~np.isnan(clean_date.data))
            clean_date = clean_date.astype("datetime64[ns]")
            cleaned[mask[0], mask[1]] = pd.DatetimeIndex(
                clean_date.data[mask[0], mask[1]]
            ).month
            sev["Cleaned"] = (("y", "x"), cleaned.astype("int16"))

        if date_filtering:
            hotspotsmask = burn_pixel.data.copy().astype("float32")
            hotspotsmask[hotspotsmask == 0] = np.nan
            firedates = (sev.StartDate.data * hotspotsmask).reshape(-1)
            values, counts = np.unique(
                firedates[~np.isnan(firedates)], return_counts=True
            )
            sortcounts = np.array(sorted(counts, reverse=True))
            datemask = np.zeros(sev.StartDate.data.shape)
            hp_masked_date = sev.StartDate * hotspotsmask.copy()
            # print(len(sortcounts))
            if len(sortcounts) <= 2:
                fireevents = sortcounts[0:1]
            else:
                fireevents = sortcounts[0:5]

            for fire in fireevents:
                # print('Change detected at: ',values[counts==fire].astype('datetime64[ns]')[0])
                firedate = values[counts == fire]

                for firei in firedate:
                    start = (
                        firei.astype("datetime64[ns]") - np.datetime64(1, "M")
                    ).astype("datetime64[ns]")
                    end = (
                        firei.astype("datetime64[ns]") - np.datetime64(-1, "M")
                    ).astype("datetime64[ns]")

                    row, col = np.where(
                        (hp_masked_date.data.astype("datetime64[ns]") >= start)
                        & (hp_masked_date.data.astype("datetime64[ns]") <= end)
                    )

                    datemask[row, col] = 1

            # burn_pixel.data = burn_pixel.data*datemask
            filtered_burnscar = burn_pixel.data.astype("float32").copy()
            filtered_burnscar = filtered_burnscar * datemask
            filtered_burnscar[filtered_burnscar == 0] = np.nan
            cleaned = np.zeros(burn_pixel.data.shape)
            clean_date = filtered_burnscar * sev.StartDate.data
            mask = np.where(~np.isnan(clean_date))
            clean_date = clean_date.astype("datetime64[ns]")
            cleaned[mask[0], mask[1]] = pd.DatetimeIndex(
                clean_date[mask[0], mask[1]]
            ).month
            sev["Cleaned"] = (("y", "x"), cleaned.astype("int16"))

    return sev


def dist_distance(params):
    """
    multiprocess version with shared memory of the cosine distances and nbr distances
    """
    ard = np.frombuffer(shared_in_arr1.get_obj(), dtype=np.int16).reshape(params[2])
    gmed = np.frombuffer(shared_in_arr2.get_obj(), dtype=np.float32).reshape(
        (params[2][0], params[2][2])
    )
    cos_dist = np.frombuffer(shared_out_arr1.get_obj(), dtype=np.float32).reshape(
        (params[2][1], params[2][2])
    )
    nbr_dist = np.frombuffer(shared_out_arr2.get_obj(), dtype=np.float32).reshape(
        (params[2][1], params[2][2])
    )
    direction = np.frombuffer(shared_out_arr3.get_obj(), dtype=np.int16).reshape(
        (params[2][1], params[2][2])
    )

    for i in range(params[0], params[1]):
        ind = np.where(ard[1, :, i] > 0)[0]

        if len(ind) > 0:
            cos_dist[ind, i] = cos_distance(gmed[:, i], ard[:, ind, i])
            nbrmed = (gmed[3, i] - gmed[5, i]) / (gmed[3, i] + gmed[5, i])
            nbr = (ard[3, :, i] - ard[5, :, i]) / (ard[3, :, i] + ard[5, :, i])
            nbr_dist[ind, i], direction[ind, i] = nbr_eucdistance(nbrmed, nbr[ind])


def dist_severity(params):
    """
    multiprocess version with shared memory of the severity algorithm
    """

    nbr = np.frombuffer(shared_in_arr01.get_obj(), dtype=np.float32).reshape(
        (-1, params[2])
    )
    nbr_dist = np.frombuffer(shared_in_arr02.get_obj(), dtype=np.float32).reshape(
        (-1, params[2])
    )
    c_dist = np.frombuffer(shared_in_arr03.get_obj(), dtype=np.float32).reshape(
        (-1, params[2])
    )
    change_dir = np.frombuffer(shared_in_arr04.get_obj(), dtype=np.int16).reshape(
        (-1, params[2])
    )
    nbr_outlier = np.frombuffer(shared_in_arr05.get_obj(), dtype=np.float32)
    cdist_outlier = np.frombuffer(shared_in_arr06.get_obj(), dtype=np.float32)
    t = np.frombuffer(shared_in_arr07.get_obj(), dtype=np.float64)

    sev = np.frombuffer(shared_out_arr01.get_obj(), dtype=np.float64)
    dates = np.frombuffer(shared_out_arr02.get_obj(), dtype=np.float64)
    days = np.frombuffer(shared_out_arr03.get_obj(), dtype=np.float64)

    for i in range(params[0], params[1]):
        sev[i], dates[i], days[i] = severity(
            nbr[:, i],
            nbr_dist[:, i],
            c_dist[:, i],
            change_dir[:, i],
            nbr_outlier[i],
            cdist_outlier[i],
            t,
            method=params[3],
        )


def distances(ard, geomed):
    """
    Calculates the cosine distance between observation and reference.
    The calculation is point based, easily adaptable to any dimension.
        Note:
            This method saves the result of the computation into the
            dists variable: p-dimensional vector with geometric
            median reflectances, where p is the number of bands.
        Args:
            ard: load from ODC
            geomed: load from odc-stats GeoMAD plugin result
            n_procs: tolerance criterion to stop iteration
    """

    n = len(ard.y) * len(ard.x)
    _x = ard

    t_dim = _x.time.data
    if len(t_dim) < 1:
        logger.warning(f"--- {len(t_dim)} observations")
        logger.warning("no enough data for the calculation of distances")
        return

    nir = _x[3, :, :, :].data.astype("float32")
    swir2 = _x[5, :, :, :].data.astype("float32")
    nir[nir <= 0] = np.nan
    swir2[swir2 <= 0] = np.nan
    nbr = (nir - swir2) / (nir + swir2)

    logger.info("begin to process distance")

    out_arr1 = mp.Array(ctypes.c_float, len(t_dim) * n)
    out_arr2 = mp.Array(ctypes.c_float, len(t_dim) * n)
    out_arr3 = mp.Array(ctypes.c_short, len(t_dim) * n)

    cos_dist = np.frombuffer(out_arr1.get_obj(), dtype=np.float32).reshape(
        (len(t_dim), n)
    )
    cos_dist.fill(np.nan)
    nbr_dist = np.frombuffer(out_arr2.get_obj(), dtype=np.float32).reshape(
        (len(t_dim), n)
    )
    nbr_dist.fill(np.nan)
    direction = np.frombuffer(out_arr3.get_obj(), dtype=np.int16).reshape(
        (len(t_dim), n)
    )
    direction.fill(0)

    in_arr1 = mp.Array(ctypes.c_short, len(ard.band) * len(_x.time) * n)
    x = np.frombuffer(in_arr1.get_obj(), dtype=np.int16).reshape(
        (len(ard.band), len(_x.time), n)
    )
    x[:] = _x.data.reshape(len(ard.band), len(_x.time), -1)

    in_arr2 = mp.Array(ctypes.c_float, len(ard.band) * n)
    gmed = np.frombuffer(in_arr2.get_obj(), dtype=np.float32).reshape(
        (len(ard.band), n)
    )
    gmed[:] = geomed.data.reshape(len(ard.band), -1)

    def init(
        shared_in_arr1_,
        shared_in_arr2_,
        shared_out_arr1_,
        shared_out_arr2_,
        shared_out_arr3_,
    ):
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

    # processes = 8
    with closing(
        mp.Pool(
            initializer=init,
            initargs=(
                in_arr1,
                in_arr2,
                out_arr1,
                out_arr2,
                out_arr3,
            ),
            processes=8,
        )
    ) as p:
        chunk = 1
        if n == 0:
            logger.warning("no point")
            return
        p.map_async(
            dist_distance,
            [(i, min(n, i + chunk), x.shape) for i in range(0, n, chunk)],
        )

    p.join()

    ds = xr.Dataset(
        coords={
            "time": t_dim,
            "y": ard.y[:],
            "x": ard.x[:],
            "band": ard.band,
        },
        attrs={"crs": "EPSG:3577"},
    )

    ds["CDist"] = (
        ("time", "y", "x"),
        cos_dist[:].reshape((len(t_dim), len(ard.y), len(ard.x))).astype("float32"),
    )
    ds["NBRDist"] = (
        ("time", "y", "x"),
        nbr_dist[:].reshape((len(t_dim), len(ard.y), len(ard.x))).astype("float32"),
    )
    ds["ChangeDir"] = (
        ("time", "y", "x"),
        direction[:].reshape((len(t_dim), len(ard.y), len(ard.x))).astype("float32"),
    )
    ds["NBR"] = (("time", "y", "x"), nbr)

    del (
        in_arr1,
        in_arr2,
        out_arr1,
        out_arr2,
        out_arr3,
        gmed,
        ard,
        cos_dist,
        nbr_dist,
        direction,
        nbr,
    )

    return ds


def outliers(dataset, distances):
    """
    Calculate the outliers for distances for change detection
    """
    logger.info("begin to process outlier")

    if distances is None:
        logger.warning("no distances for the outlier calculations")
        return
    nbr_ps = nanpercentile(distances.NBRDist.data, [25, 75])

    nbr_outlier = nbr_ps[1] + 1.5 * (nbr_ps[1] - nbr_ps[0])
    cos_distps = nanpercentile(distances.CDist.data, [25, 75])
    cos_dist_outlier = cos_distps[1] + 1.5 * (cos_distps[1] - cos_distps[0])

    ds = xr.Dataset(
        coords={"y": dataset.y[:], "x": dataset.x[:]}, attrs={"crs": "EPSG:3577"}
    )
    ds["CDistoutlier"] = (("y", "x"), cos_dist_outlier.astype("float32"))
    ds["NBRoutlier"] = (("y", "x"), nbr_outlier.astype("float32"))
    return ds


def region_growing(severity, dists, outlrs):
    """
    The function includes further areas that do not qualify as outliers but do show a substantial decrease in NBR and
    are adjoining pixels detected as burns. These pixels are classified as 'moderate severity burns'.
        Note: this function is build inside the 'severity' function
            Args:
                severity: xarray including severity and start-date of the fire
    """
    start_date = severity.StartDate.data[~np.isnan(severity.StartDate.data)].astype(
        "datetime64[ns]"
    )
    change_dates = np.unique(start_date)
    z_distance = 2 / 3  # times outlier distance (eq. 3 stdev)

    from scipy import ndimage
    from skimage import measure

    # see http://www.scipy-lectures.org/packages/scikit-image/index.html#binary-segmentation-foreground-background
    fraction_seedmap = 0.25  # this much of region must already have been mapped as burnt to be included
    seed_map = (severity.Severe.data > 0).astype(
        int
    )  # use 'Severe' burns as seed map to grow

    start_dates = np.zeros((len(dists.y), len(dists.x)))
    start_dates[~np.isnan(severity.StartDate.data)] = start_date
    tmp_map = seed_map
    annual_map = seed_map
    # grow the region based on StartDate
    for d in change_dates:

        di = str(d)[:10]
        ti = np.where(dists.time > np.datetime64(di))[0][0]
        nbr_score = (dists.ChangeDir * dists.NBRDist)[ti, :, :] / outlrs.NBRoutlier
        cos_score = (dists.ChangeDir * dists.CDist)[ti, :, :] / outlrs.CDistoutlier
        potential = ((nbr_score > z_distance) & (cos_score > z_distance)).astype(int)
        # Use the following line if using NBR is preferred
        # Potential = ((dists.NBR[ti, :, :] > 0) & (cos_score > z_distance)).astype(int)

        all_labels = measure.label(
            potential.astype(int).values, background=0
        )  # labelled all the conneted component
        new_potential = ndimage.mean(seed_map, labels=all_labels, index=all_labels)
        new_potential[all_labels == 0] = 0

        annual_map = annual_map + (new_potential > fraction_seedmap).astype(int)
        annual_map = (annual_map > 0).astype(int)
        start_dates[
            (annual_map - tmp_map) > 0
        ] = d  # assign the same date to the growth region
        tmp_map = annual_map

    burn_extent = annual_map

    return burn_extent, start_dates


def _create_geospatial_attributes(dataset):
    """Creates geospatial attributes for the dataset
    Input: dataset
    Returns: lat_max, lat_min, lon_max, lon_min, poly
    which are the corners of the polygon and the polygon to use
    """
    import pyproj

    y_max, y_min, x_max, x_min = (
        np.max(dataset.y),
        np.min(dataset.y),
        np.max(dataset.x),
        np.min(dataset.x),
    )

    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3577")
    [lon_max, lon_min], [lat_max, lat_min] = transformer.transform(
        [y_max, y_min], [x_max, x_min]
    )

    # get the corners fot the polygon
    p1 = geometry.Point(lon_min, lat_min)
    p2 = geometry.Point(lon_max, lat_min)
    p3 = geometry.Point(lon_max, lat_max)
    p4 = geometry.Point(lon_min, lat_max)

    # make the polygon
    point_list = [p1, p2, p3, p4, p1]
    poly = geometry.Polygon([[p.x, p.y] for p in point_list])
    return lat_max, lat_min, lon_max, lon_min, poly


def _create_variable_attributes(dataset):
    """puts variable attributes into the dataset"""
    dataset["y"].attrs = {
        "units": "metre",
        "long_name": "y coordinate of projection",
        "standard_name": "projection_y_coordinate",
    }
    dataset["x"].attrs = {
        "units": "metre",
        "long_name": "x coordinate of projection",
        "standard_name": "projection_x_coordinate",
    }
    dataset["StartDate"].attrs = {
        "long_name": "StartDate",
        "standard_name": "StartDate",
        "axis": "T",
        "coverage_content_type": "model results",
    }
    dataset["Duration"].attrs = {
        "units": "number of days",
        "long_name": "Duration",
        "standard_name": "Duration",
        "coverage_content_type": "model results",
    }
    dataset["Severity"].attrs = {
        "units": "temporal integral of cosine distance for the duration of detected severe burned",
        "long_name": "Severity",
        "standard_name": "Severity",
        "coverage_content_type": "model results",
    }
    dataset["Severe"].attrs = {
        "units": "1",
        "long_name": "Severe burned area",
        "standard_name": "Severe",
        "coverage_content_type": "model results",
    }
    dataset["Moderate"].attrs = {
        "units": "1",
        "long_name": "Moderate burned area",
        "standard_name": "Moderate",
        "coverage_content_type": "model results",
    }
    dataset["Corroborate"].attrs = {
        "units": "1",
        "long_name": "Corrorborate evidence",
        "standard_name": "Corroborate",
        "coverage_content_type": "model results",
    }
    dataset["Cleaned"].attrs = {
        "units": "month",
        "long_name": "Cleaned",
        "standard_name": "Cleaned",
        "coverage_content_type": "model results",
    }
    return dataset


def create_attributes(dataset, product_name, version, method, res=30):
    """creates attributes for the dataset so that it will have information when output"""
    import datetime

    dataset = _create_variable_attributes(dataset)
    lat_max, lat_min, lon_max, lon_min, poly = _create_geospatial_attributes(dataset)
    product_definition = {
        "name": product_name,
        "description": "Description for " + product_name,
        "mapping_Method": method,  # NBR or NBRdist
        "data_source": "Landsat 5/8",
        "resolution": str(res) + " m",
        "cdm_data_type": "Grid",
        "cmi_id": "BAM_25_1.0",
        "Conventions": "CF-1.6, ACDD-1.3",
        "geospatial_bounds": str(poly),
        "geospatial_bounds_crs": "EPSG:4326",
        "geospatial_lat_max": lat_max,
        "geospatial_lat_min": lat_min,
        "geospatial_lat_units": "degree_north",
        "geospatial_lon_max": lon_max,
        "geospatial_lon_min": lon_min,
        "geospatial_lon_units": "degree_east",
        "date_created": datetime.datetime.today().isoformat(),
        "history": "NetCDF-CF file created by datacube version '1.6rc2+108.g096a26d5' at 20180914",
        "institution": "Commonwealth of Australia (Geoscience Australia)",
        "keywords": "AU/GA,NASA/GSFC/SED/ESD/LANDSAT,ETM+,TM,OLI,EARTH SCIENCE, BURNED AREA",
        "keywords_vocabulary": "GCMD",
        "license": "CC BY Attribution 4.0 International License",
        "product_version": version,
        "publisher_email": "earth.observation@ga.gov.au",
        "publisher_url": "http://www.ga.gov.au",
        "references": "Renzullo et al (2019)",
        "source": "Burned Area Map Algorithm v1.0",
        "summary": "",
        "title": "Burned Area Map Annual 1.0",
    }
    for key, value in product_definition.items():
        dataset.attrs[key] = value

    return dataset


def hotspot_polygon(period, extent, buffersize, hotspotfile):
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

    logger.info("begin to process hotspot polygon")

    # print("extent", extent)

    # year = int(str(period[0])[0:4])
    # if year >= 2019:
    #    logger.warning("No complete hotspots data after 2018")
    #    return None

    _ = s3fs.S3FileSystem(anon=True)

    # hotspotfile = (
    #    "s3://dea-public-data-dev/projects/burn_cube/ancillary_file/hotspot_historic.csv"
    # )

    # if os.path.isfile(hotspotfile):
    #    column_names = ["datetime", "sensor", "latitude", "longitude"]
    #    table = pd.read_csv(hotspotfile, usecols=column_names)
    # else:
    #    logger.warning("No hotspots file is found")
    #    return None

    column_names = ["datetime", "sensor", "latitude", "longitude"]
    table = pd.read_csv(hotspotfile, usecols=column_names)

    start = (
        np.datetime64(period[0]).astype("datetime64[ns]") - np.datetime64(2, "M")
    ).astype("datetime64[ns]")
    stop = np.datetime64(period[1])
    extent[0] = extent[0] - 100000
    extent[1] = extent[1] + 100000
    extent[2] = extent[2] - 100000
    extent[3] = extent[3] + 100000

    dates = pd.to_datetime(table.datetime.apply(lambda x: x.split("+")[0]).values)

    transformer = pyproj.Transformer.from_crs("EPSG:3577", "EPSG:4283")
    lat, lon = transformer.transform(extent[0:2], extent[2:4])

    index = np.where(
        (table.sensor == "MODIS")
        & (dates >= start)
        & (dates <= stop)
        & (table.latitude <= lat[1])
        & (table.latitude >= lat[0])
        & (table.longitude <= lon[1])
        & (table.longitude >= lon[0])
    )[0]

    latitude = table.latitude.values[index]
    longitude = table.longitude.values[index]

    reverse_transformer = pyproj.Transformer.from_crs("EPSG:4283", "EPSG:3577")
    easting, northing = reverse_transformer.transform(latitude, longitude)

    patch = [
        Point(easting[i], northing[i]).buffer(buffersize) for i in range(0, len(index))
    ]
    polygons = unary_union(patch)

    return polygons


def severitymapping(
    dists, outlrs, period, hotspotfile, method="NBR", growing=True, hotspots_period=None
):
    """Calculates burnt area for a given period
    Args:
        period: period of time with burn mapping interest,  e.g.('2015-01-01','2015-12-31')
        n_procs: tolerance criterion to stop iteration
        method: methods for change detection
        growing: whether to grow the region
    """

    logger.info("begin to process severity")

    if dists is None:
        logger.warning("no data available for severity mapping")
        return None

    c_dist = dists.CDist.data.reshape((len(dists.time), -1))
    cdist_outlier = outlrs.CDistoutlier.data.reshape(len(dists.x) * len(dists.y))
    nbr_dist = dists.NBRDist.data.reshape((len(dists.time), -1))
    nbr = dists.NBR.data.reshape((len(dists.time), -1))
    nbr_outlier = outlrs.NBRoutlier.data.reshape(len(dists.x) * len(dists.y))
    change_dir = dists.ChangeDir.data.reshape((len(dists.time), -1))

    if method == "NBR":
        tmp = (
            dists.CDist.where((dists.CDist > outlrs.CDistoutlier) & (dists.NBR < 0))
            .sum(axis=0)
            .data
        )
        tmp = tmp.reshape(len(dists.x) * len(dists.y))
        outlierind = np.where(tmp > 0)[0]

    elif method == "NBRdist":
        tmp = (
            dists.CDist.where(
                (dists.CDist > outlrs.CDistoutlier)
                & (dists.NBRDist > outlrs.NBRoutlier)
                & (dists.ChangeDir == 1)
            )
            .sum(axis=0)
            .data
        )
        tmp = tmp.reshape(len(dists.x) * len(dists.y))
        outlierind = np.where(tmp > 0)[0]

    else:
        raise ValueError

    if len(outlierind) == 0:
        logger.warning("no burnt area detected")
        return None
    # input shared arrays
    in_arr1 = mp.Array(ctypes.c_float, len(dists.time[:]) * len(outlierind))
    nbr_shared = np.frombuffer(in_arr1.get_obj(), dtype=np.float32).reshape(
        (len(dists.time[:]), len(outlierind))
    )
    nbr_shared[:] = nbr[:, outlierind]

    in_arr2 = mp.Array(ctypes.c_float, len(dists.time[:]) * len(outlierind))
    nbr_dist_shared = np.frombuffer(in_arr2.get_obj(), dtype=np.float32).reshape(
        (len(dists.time[:]), len(outlierind))
    )
    nbr_dist_shared[:] = nbr_dist[:, outlierind]

    in_arr3 = mp.Array(ctypes.c_float, len(dists.time[:]) * len(outlierind))
    cosdist_shared = np.frombuffer(in_arr3.get_obj(), dtype=np.float32).reshape(
        (len(dists.time[:]), len(outlierind))
    )
    cosdist_shared[:] = c_dist[:, outlierind]

    in_arr4 = mp.Array(ctypes.c_short, len(dists.time[:]) * len(outlierind))
    change_dir_shared = np.frombuffer(in_arr4.get_obj(), dtype=np.int16).reshape(
        (len(dists.time[:]), len(outlierind))
    )
    change_dir_shared[:] = change_dir[:, outlierind]

    in_arr5 = mp.Array(ctypes.c_float, len(outlierind))
    nbr_outlier_shared = np.frombuffer(in_arr5.get_obj(), dtype=np.float32)
    nbr_outlier_shared[:] = nbr_outlier[outlierind]

    in_arr6 = mp.Array(ctypes.c_float, len(outlierind))
    cdist_outlier_shared = np.frombuffer(in_arr6.get_obj(), dtype=np.float32)
    cdist_outlier_shared[:] = cdist_outlier[outlierind]

    in_arr7 = mp.Array(ctypes.c_double, len(dists.time[:]))
    t = np.frombuffer(in_arr7.get_obj(), dtype=np.float64)
    t[:] = dists.time.data.astype("float64")

    # output shared arrays
    out_arr1 = mp.Array(ctypes.c_double, len(outlierind))
    sev = np.frombuffer(out_arr1.get_obj(), dtype=np.float64)
    sev.fill(np.nan)

    out_arr2 = mp.Array(ctypes.c_double, len(outlierind))
    dates = np.frombuffer(out_arr2.get_obj(), dtype=np.float64)
    dates.fill(np.nan)

    out_arr3 = mp.Array(ctypes.c_double, len(outlierind))
    days = np.frombuffer(out_arr3.get_obj(), dtype=np.float64)
    days.fill(0)

    def init(
        shared_in_arr1_,
        shared_in_arr2_,
        shared_in_arr3_,
        shared_in_arr4_,
        shared_in_arr5_,
        shared_in_arr6_,
        shared_in_arr7_,
        shared_out_arr1_,
        shared_out_arr2_,
        shared_out_arr3_,
    ):
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

    with closing(
        mp.Pool(
            initializer=init,
            initargs=(
                in_arr1,
                in_arr2,
                in_arr3,
                in_arr4,
                in_arr5,
                in_arr6,
                in_arr7,
                out_arr1,
                out_arr2,
                out_arr3,
            ),
            processes=8,
        )
    ) as p:
        chunk = 1
        if len(outlierind) == 0:
            return
        p.map_async(
            dist_severity,
            [
                (i, min(len(outlierind), i + chunk), len(outlierind), method)
                for i in range(0, len(outlierind), chunk)
            ],
        )

    p.join()

    sevindex = np.zeros(len(dists.y) * len(dists.x))
    duration = np.zeros(len(dists.y) * len(dists.x)) * np.nan
    startdate = np.zeros(len(dists.y) * len(dists.x)) * np.nan
    sevindex[outlierind] = sev
    duration[outlierind] = days
    startdate[outlierind] = dates
    sevindex = sevindex.reshape((len(dists.y), len(dists.x)))
    duration = duration.reshape((len(dists.y), len(dists.x)))
    startdate = startdate.reshape((len(dists.y), len(dists.x)))
    startdate[startdate == 0] = np.nan
    duration[duration == 0] = np.nan
    del (
        in_arr1,
        in_arr2,
        in_arr3,
        in_arr4,
        in_arr5,
        in_arr6,
        out_arr1,
        out_arr2,
        out_arr3,
    )
    del (
        sev,
        days,
        dates,
        nbr_shared,
        nbr_dist_shared,
        cosdist_shared,
        nbr_outlier_shared,
        change_dir_shared,
        cdist_outlier_shared,
    )

    out = xr.Dataset(coords={"y": dists.y[:], "x": dists.x[:]})
    out["StartDate"] = (("y", "x"), startdate)
    out["Duration"] = (("y", "x"), duration.astype("int16"))
    burnt = np.zeros((len(dists.y), len(dists.x)))
    burnt[duration > 0] = 1
    out["Severity"] = (("y", "x"), sevindex.astype("float32"))
    out["Severe"] = (("y", "x"), burnt.astype("int16"))

    if burnt.sum() == 0:
        out["Corroborate"] = (
            ("y", "x"),
            np.zeros((len(dists.y), len(dists.x))).astype("int16"),
        )
        out["Moderate"] = (
            ("y", "x"),
            np.zeros((len(dists.y), len(dists.x))).astype("int16"),
        )
        out["Cleaned"] = (
            ("y", "x"),
            np.zeros((len(dists.y), len(dists.x))).astype("int16"),
        )
        return create_attributes(out, "Burned Area Map", "v1.0", method)

    if growing:
        burn_area, growing_dates = region_growing(out, dists, outlrs)
        out["Moderate"] = (("y", "x"), burn_area.astype("int16"))
        growing_dates[growing_dates == 0] = np.nan
        out["StartDate"] = (("y", "x"), growing_dates)

    extent = [
        np.min(dists.x.data),
        np.max(dists.x.data),
        np.min(dists.y.data),
        np.max(dists.y.data),
    ]

    if hotspots_period is None:
        hotspots_period = period
    polygons = hotspot_polygon(
        hotspots_period, extent, 4000, hotspotfile
    )  # generate hotspot polygons with 4km buffer

    # default mask
    hot_spot_mask = np.zeros((len(dists.y), len(dists.x)))

    if polygons is None or polygons.is_empty:
        logger.warning("no hotspots data")
    else:
        coords = out.coords

        if polygons.type == "MultiPolygon":
            for polygon in polygons.geoms:
                hot_spot_mask_tmp = outline_to_mask(
                    polygon.exterior, coords["x"], coords["y"]
                )
                hot_spot_mask = hot_spot_mask_tmp + hot_spot_mask
            hot_spot_mask = xr.DataArray(hot_spot_mask, coords=coords, dims=("y", "x"))
        if polygons.type == "Polygon":
            hot_spot_mask = outline_to_mask(polygons.exterior, coords["x"], coords["y"])
            hot_spot_mask = xr.DataArray(hot_spot_mask, coords=coords, dims=("y", "x"))

        out["Corroborate"] = (("y", "x"), hot_spot_mask.data.astype("int16"))
        out = post_filtering(out, hotspots_filtering=True, date_filtering=False)
    return create_attributes(out, "Burned Area Map", "v1.0", method)


def burnpixel_masking(data, varname):
    """
    This function converts the severity map into a burn pixel mask
    Required input:
    data: severity data in 2D, e.g. output from severity_mapping in the changedection.py
    """
    burnpixel = data[varname]
    burnpixel.data[burnpixel.data > 1] = 1
    return burnpixel
