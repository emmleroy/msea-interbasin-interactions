"""
SEAM Project Utility Functions
===================================================================

A collection of useful functions for climate data analysis with a
focus on modes of climate variability, teleconnections, monsoon
rainfall.

"""

from typing import Set, Union, Pattern, List
from scipy.signal import detrend
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy import stats
from seam import cesm_utils
import xskillscore as xs
import os
import regionmask

supported_fonts: List[str] = {'Andale Mono', 'Arial', 'Arial Black',
                               'Comic Sans MS', 'Courier New', 'Georgia',
                               'Impact', 'Times New Roman', 'Trebuchet MS',
                               'Verdana', 'Webdings', 'Amiri', 'Lato', 'Roboto', 'Futura'}


def set_matplotlib_font(font_family: str):
    """Set the matplotlib font family.

    Args:
        font_family (str): the font family

    """

    #assert font_family in supported_fonts, f'Font {font_family} not supported.'
    plt.rcParams['font.family'] = font_family
    plt.rcParams.update({'mathtext.default': 'regular'})
def crop_da(da, minlon, maxlon, minlat, maxlat):
    cropped_da = da.sel(
        lat=slice(minlat, maxlat),
        lon=slice(minlon, maxlon),
    )
    return cropped_da


def get_file_list(directory_path: str, compiled_regex: Pattern):
    """Return a list of file paths in a directory matching the regex pattern.
    Args:
        directory_path (str): path to directory where to look for files
        compiled_regex (pattern): compiled regex pattern to match
    Returns:
        file_list (list[str]): list of file paths
    """

    file_list = []
    for file_name in os.listdir(directory_path):
        if compiled_regex.match(file_name):
            file_list.append(os.path.join(directory_path, file_name))

    # Important!!! Sort to concatenate chronologically
    file_list.sort()

    return file_list


def get_ds(file_path: str, cesm=False):
    """Open geospatial data and convert to the following conventions:

    - rename "latitude/longitude" coordinates to "lat/lon"
    - convert decreasing lats (90 to -90) to increasing (-90 to 90)
    - convert negative lons (-180 to 180) to positive (0-360)

    Args:
        file_path (str) : full path to geospatial data

    """

    ds = xr.open_dataset(file_path)

    # Check name of coordinates:
    if 'latitude' in ds.dims and 'longitude' in ds.dims:
        ds = ds.rename({'longitude': 'lon', 'latitude': 'lat'})
    elif 'nlat' in ds.dims and 'nlon' in ds.dims:
        ds = cesm_utils.regrid_cesm(
                    ds,
                    d_lon_lat_kws={"lon": 1, "lat": 1},
                    method="bilinear",
                    periodic=False,
                    filename=None,
                    reuse_weights=None,
                    tsmooth_kws=None,
                    how=None,
                )
    if 'pre' in ds.keys():
        ds = ds.rename({'pre': 'precip'})
    if cesm:
        ds = cesm_utils.time_set_midmonth(ds, 'time', deep=False)
    # Check if latitudes are monotonically increasing:
    if np.all(np.diff(ds.lat) < 0):
        ds = ds.reindex(lat=list(reversed(ds['lat'])))

    # Check for negative longitude values:
    if any(x < 0 for x in ds.lon):
        ds.coords['lon'] = (ds.coords['lon'] % 360)
        #TODO: Check if I need to reindex?
        ds = ds.sortby(ds.lon)

    return ds


def get_multifile_ds_cesm(file_list: List[str], variable, chunks='auto'):
    """Open multiple files as a single dataset.
    WARNING: compatibility checks are overrided for speed-up!
    Args:
        file_list (list(str): list of file paths
    Returns:
        ds (xr.DataSet): a single concatenated xr.Dataset
    """

    v = xr.__version__.split(".")

    if int(v[0]) == 0 and int(v[1]) >= 15:
        ds = xr.open_mfdataset(file_list,
                               combine='nested',
                               concat_dim='time',
                               chunks=chunks,
                               parallel=True,
                               data_vars='minimal',
                               coords='minimal',
                               compat='override')
    else:
        ds = xr.open_mfdataset(file_list,
                              combine='nested',
                               concat_dim='time',
                               parallel=True,
                               data_vars='minimal',
                               coords='minimal',
                               compat='override',
                              engine='netcdf4')

    ds = ds[variable].to_dataset()

    # Check name of coordinates:
    if 'latitude' in ds.dims and 'longitude' in ds.dims:
        ds = ds.rename({'longitude': 'lon', 'latitude': 'lat'})
    elif 'nlat' in ds.dims and 'nlon' in ds.dims:
        ds = cesm_utils.regrid_cesm(
                    ds,
                    d_lon_lat_kws={"lon": 1, "lat": 1},
                    method="bilinear",
                    periodic=False,
                    filename=None,
                    reuse_weights=None,
                    tsmooth_kws=None,
                    how=None,
                )
    if 'pre' in ds.keys():
        ds = ds.rename({'pre': 'precip'})
    ds = cesm_utils.time_set_midmonth(ds, 'time', deep=False)
    # Check if latitudes are monotonically increasing:
    if np.all(np.diff(ds.lat) < 0):
        ds = ds.reindex(lat=list(reversed(ds['lat'])))

    # Check for negative longitude values:
    if any(x < 0 for x in ds.lon):
        ds.coords['lon'] = (ds.coords['lon'] % 360)
        #TODO: Check if I need to reindex?
        ds = ds.sortby(ds.lon)
        
    return ds


def get_multifile_ds(file_list: List[str], cesm=False, chunks='auto'):
    """Open multiple files as a single dataset.
    WARNING: compatibility checks are overrided for speed-up!
    Args:
        file_list (list(str): list of file paths
    Returns:
        ds (xr.DataSet): a single concatenated xr.Dataset
    """

    v = xr.__version__.split(".")

    if int(v[0]) == 0 and int(v[1]) >= 15:
        ds = xr.open_mfdataset(file_list,
                               combine='nested',
                               concat_dim='time',
                               chunks=chunks,
                               parallel=True,
                               data_vars='minimal',
                               coords='minimal',
                               compat='override')
    else:
        ds = xr.open_mfdataset(file_list,
                              combine='nested',
                               concat_dim='time',
                               parallel=True,
                               data_vars='minimal',
                               coords='minimal',
                               compat='override',
                              engine='netcdf4')

    # Check name of coordinates:
    if 'latitude' in ds.dims and 'longitude' in ds.dims:
        ds = ds.rename({'longitude': 'lon', 'latitude': 'lat'})
    elif 'nlat' in ds.dims and 'nlon' in ds.dims:
        ds = cesm_utils.regrid_cesm(
                    ds,
                    d_lon_lat_kws={"lon": 1, "lat": 1},
                    method="bilinear",
                    periodic=False,
                    filename=None,
                    reuse_weights=None,
                    tsmooth_kws=None,
                    how=None,
                )
    if 'pre' in ds.keys():
        ds = ds.rename({'pre': 'precip'})
    if cesm:
        ds = cesm_utils.time_set_midmonth(ds, 'time', deep=False)
    # Check if latitudes are monotonically increasing:
    if np.all(np.diff(ds.lat) < 0):
        ds = ds.reindex(lat=list(reversed(ds['lat'])))

    # Check for negative longitude values:
    if any(x < 0 for x in ds.lon):
        ds.coords['lon'] = (ds.coords['lon'] % 360)
        #TODO: Check if I need to reindex?
        ds = ds.sortby(ds.lon)
        
    return ds


def check_data_conventions(ds: Union[xr.Dataset, xr.DataArray]):
    """Check if data conventions are satisfied.

    - "latitude/longitude" coords must be "lat/lon"
    - latitudes must be monotonically increasing (-90 to 90)
    - longitudes must be positive (0-360)

    Args:
        ds (xr.Dataset or xr.DataArray) : data to check

    """
    if 'latitude' in ds.dims and 'longitude' in ds.dims:
        raise NotImplementedError(
            """Change 'longitude' to 'lon' and 'latitude' to 'lat'.
               Try using seam.utils.get_ds().""")
    if np.all(np.diff(ds.lat) < 0):
        raise NotImplementedError(
            """Latitudes must be monotonically increasing.
               Try using seam.utils.get_ds().""")

    if any(x < 0 for x in ds.lon):
        raise NotImplementedError(
            """Longitudes must range from 0 to 360°.
               Try using seam.utils.get_ds().""")


def calc_cos_wmean(da: xr.DataArray):
    """Calculate the spatial mean weighted by the cosine of the latitudes.

    The correction version of da.mean(dim=['lat', 'lon']) considering that
    lat-lon grid boxes occupy less area from the topics to the poles (i.e.
    data near the equator should have more influence on the mean as it occupies
    more area).

    Args:
        da (xr.DataArray): data with cartesian coordinates to average

    Returns:
        weighted_mean (xr.DataArray): area-weighted spatial mean

    """
    check_data_conventions(da)

    weights = np.cos(np.deg2rad(da.lat))
    weights.name = "weights"

    area_weighted = da.weighted(weights)
    weighted_mean = area_weighted.mean(('lat', 'lon'), keep_attrs=True)

    return weighted_mean


def detrend_array(da: xr.DataArray, dim: str, deg=1):
    """Detrend with polyfit along a single dimension.

    Copied from Ryan Abernathey at:
    https://gist.github.com/rabernat/1ea82bb067c3273a6166d1b1f77d490f

    Args:
        da (xr.DataArray): the data to detrend
        dim (str): dimension along which to apply detrend
        deg (int): polyfit degree (default deg=1 for linear detrending)

    Returns:
        da - fit (xr.DataArray): the detrended data

    """
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)

    return (da - fit) + da.mean(dim='time')


def remove_monthly_clm(da: xr.DataArray, base_start: str, base_end: str):
    """Subtract the monthly climatology from each grid cell to remove
     the seasonal cycle and compute monthly mean anomalies

     Note: NEW! base start and end month must always be specified

    Args:
        da (xr.DataArray) : input of monthly data
        base_start (str)  : start year and month of base period (i.e. '1951-01')
        base_end (str)    : end year and month of base period (i.e. '2015-12')

    Returns:
        monthly_mean_anm (xr.DataArray): monthly mean anomalies

    """

    base_da = da.sel(time=slice(base_start, base_end))
    monthly_mean_clm = base_da.groupby("time.month").mean(dim="time")

    monthly_mean_anm = da.groupby("time.month") - monthly_mean_clm
    
    return monthly_mean_anm


def mask_ocean(da):
    """Mask ocean areas"""
    land = regionmask.defined_regions.natural_earth_v5_0_0.land_110
    mask = land.mask(da["lon"], da["lat"])
    da_new = da.where(~np.isnan(mask))
    return da_new
    

def get_percent_anm(da: xr.DataArray):
    """Subtract the monthly climatology from each grid cell to remove
     the seasonal cycle and compute monthly mean anomalies

     Note: the default climatological period here is the entire time period 
     of the input data array.

    Args:
        da (xr.DataArray) : input of monthly data

    Returns:
        monthly_mean_anm (xr.DataArray): monthly mean anomalies

    """

    monthly_mean_clm = da.groupby("time.month").mean(dim="time")

    monthly_mean_anm = da.groupby("time.month") - monthly_mean_clm
    
    percent = (monthly_mean_anm.groupby("time.month") / monthly_mean_clm)*100

    return percent


def autocorr(x, t=1):
    """Calculates autocorrelation with lag = 1.
    
    Parameters
    ----------
    x: numpy.array
        Input
    
    Returns
    -------
    ac: numpy.float64
        Autocorrelation with lag = 1
    """
    ac = np.corrcoef(np.array([x[0:len(x)-t], x[t:len(x)]]))[0][1]
    return ac


def df_eff(N, ts1, ts2, t=1):
    """ Determines effective degrees of freedom
    from eq. 31 in (Bretherton et al, 1999).
    N = sample size and is typically the size of ts1 (and ts2)."""
    rx = autocorr(ts1, t=t)
    ry = autocorr(ts2, t=t)
    df_eff = N*((1-(rx*ry))/(1+(rx*ry)))-2 
    return df_eff


def ttest_1samp(a, popmean, dim):
    """
    TO DO: add documentation
    """
    t_val, p_val = stats.ttest_1samp(a, popmean, axis=a.get_axis_num(dim))

    return t_val, p_val


def mc_ttest_1samp(a, popmean, dim, reps):
    """
    1-sample t-test with resampling
    """

    # Initialize random seed to get the same 'random' numbers each time
    np.random.seed(4578930)

    # Resample the data with replacement along given dimension 'dim'
    resampled_a = xs.resample_iterations_idx(a, reps, dim, replace=True)

    # Get axis number for given dimension
    axis_num =a.get_axis_num(dim)

    # Calculate p-values for each iteration of the resampled data
    sim_pvals = []
    for idx in range(reps):
        sample = resampled_a.sel(iteration=idx)
        _, resample_pval = ttest_1samp(sample, popmean, dim)
        sim_pvals.append(resample_pval)
    sim_pvals = np.stack(sim_pvals)

    # Calculate the observed p-value
    _, obs_pval = ttest_1samp(a, popmean, dim)

    # Estimate p-value as probability that simulated p-values are more
    # extreme than the observed p-value
    total_probability = np.sum(sim_pvals <= obs_pval, axis=axis_num)/reps

    return total_probability


def mc_test_confidence_interval(a, popmean, dim, reps):
    """
    1-sample t-test with resampling
    """

    # Initialize random seed to get the same 'random' numbers each time
    np.random.seed(4578930)

    # Resample the data with replacement along given dimension 'dim'
    resampled_a = xs.resample_iterations_idx(a, reps, dim, replace=True)

    # Get axis number for given dimension
    axis_num =a.get_axis_num(dim)

    # Calculate p-values for each iteration of the resampled data
    sim_pvals = []
    for idx in range(reps):
        sample = resampled_a.sel(iteration=idx)
        _, resample_pval = ttest_1samp(sample, popmean, dim)
        sim_pvals.append(resample_pval)
    sim_pvals = np.stack(sim_pvals)

    # Calculate the observed p-value
    _, obs_pval = ttest_1samp(a, popmean, dim)

    # Estimate p-value as probability that simulated p-values are more
    # extreme than the observed p-value
    total_probability = np.sum(sim_pvals <= obs_pval, axis=axis_num)/reps

    return total_probability


def mc_test_mean_diff(a, dim, reps, lower_quantile, upper_quantile):
    """
    TO DO: add docs
    """
    np.random.seed(4578930)
    reps = 10000
    resampled_data = xs.resample_iterations_idx(a, reps, dim, replace=True)
    resampled_means = resampled_data.mean(dim='time')
    pct_upper = resampled_means.quantile(upper_quantile, dim='iteration')
    pct_lower = resampled_means.quantile(lower_quantile, dim='iteration')

    a_mean = a.mean(dim='time')
    a_notsig = a_mean.where(pct_lower<=0).where(0<=pct_upper)

    return a_notsig