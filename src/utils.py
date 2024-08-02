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
import pandas as pd
from scipy import stats
from seam import cesm_utils, precip, nino34
import xskillscore as xs
import os
import regionmask

# DEFINE DIRECTORIES HERE
GPCC_DIR = "/home/eleroy/proj-dirs/SEAM/data/ExtData/GPCC/full_v2020/"
CRUT_DIR = "/home/eleroy/proj-dirs/SEAM/data/ExtData/CRU_TS4.06/"
APHR_DIR = "/home/eleroy/proj-dirs/SEAM/data/ExtData/APHRODITE/"

ERSST_DIR = "/home/eleroy/proj-dirs/SEAM/data/ExtData/ERSST/"
HADIS_DIR = "/home/eleroy/proj-dirs/SEAM/data/ExtData/HadISST/"
COBES_DIR = "/home/eleroy/proj-dirs/SEAM/data/ExtData/COBE_SST2/"


supported_fonts: List[str] = {'Andale Mono', 'Arial', 'Arial Black',
                               'Comic Sans MS', 'Courier New', 'Georgia',
                               'Impact', 'Times New Roman', 'Trebuchet MS',
                               'Verdana', 'Webdings', 'Amiri', 'Lato', 'Roboto', 'Futura'}


### Figure 1 ###
def get_running_corr(array1, array2, window=13, min_periods=5, center=True):
    """Apply a rolling correlation coefficient"""
    s1 = pd.Series(array1)
    s2 = pd.Series(array2)
    corr = s1.rolling(window, min_periods=min_periods, center=center).corr(s2)
    ds = xr.Dataset({"corr": corr.values})
    ds["time"] = array1.time
    clean_ds = ds.reset_index("corr").reset_coords()
    return clean_ds


def get_obs_precip_anomalies(source, months, detrend=False):
    if source == "GPCC":
        file = f'{GPCC_DIR}/precip.mon.total.0.5x0.5.v2020.nc'
    elif source == "CRUT":
        file = f'{CRUT_DIR}/cru_ts4.06.1901.2021.pre.dat.nc'
    elif source == "APHR":
        file = f'{APHR_DIR}/APHRO_MA_050deg_V1101_EXR1.1951-2015.mm_per_month.nc'
    else: 
        raise NotImplementedError(
            """Source must be one of "GPCC", "CRUT", "APHRO" """)
    ds0 = get_ds(file)
    precip_ds = ds0.sel(time=slice("1951-01", "2015-12"))
    precip_da = precip_ds["precip"]
    precip_anm = (
        precip.get_SEAM_anm_timeseries(
            precip_da,
            detrend=detrend,
            base_start='1951-01',
            base_end='2015-12',
            monsoon_season=False,
            monthly=True,
        ))

    precip_MAM = precip_anm.sel(time=precip_anm.time.dt.season=="MAM")
    precip_anm = precip_MAM.resample(time="1Y").mean()
    return precip_anm


def get_obs_nino34_sst_anomalies(source, detrend=False):
    if source == "ERSST":
            file = f'{ERSST_DIR}/sst.mnmean.v5.nc'
    elif source == "HADISST":
        file = f'{HADIS_DIR}/HadISST_sst.nc'
    elif source == "COBESST":
        file = f'{COBES_DIR}/sst.mon.mean.nc'
    else: 
        raise NotImplementedError(
            """Source must be one of "ERSST", "HADISST", "COBESST" """)
    ds0 = get_ds(file)
    sst_ds = ds0.sel(time=slice("1951-01", "2015-12"))
    sst_da = sst_ds["sst"]
    sst_anm_nino34_ersst = nino34.get_nino34_anm_timeseries(
        sst_da, detrend=detrend, base_start='1951-01',
        base_end='2015-12', filtered=True
    )
    sst_season = (
        sst_anm_nino34_ersst.resample(time="QS-DEC", label="left")
        .mean(dim='time')
        .sel(time=slice("1951-01", "2015-12"))
    )  # take quarterly means starting Dec 1
    nino34_DJF_ersst = (
        sst_season.sel(time=sst_season.time.dt.month.isin([12]))
        .resample(time="1Y")
        .mean()
    )
    return nino34_DJF_ersst


def get_model_precip_anomalies(ds, months, detrend=False):
    precip_ds = ds.sel(time=slice("1900-01", "2100-12"))
    precip_da = precip_ds["PRECT"]
    precip_anm = (
        precip.get_SEAM_anm_timeseries(
            precip_da,
            detrend=detrend,
            base_start='1951-01',
            base_end='2015-12',
            monsoon_season=False,
            monthly=True,
        ))
    precip_anm = precip_anm.sel(time=precip_anm.time.dt.month.isin(months))

    # Convert m/s to mm/month 
    # Seconds per month (non-leap years in CESM calendar)
    seconds_per_month = {
        3: 31 * 24 * 60 * 60,  # March
        4: 30 * 24 * 60 * 60,  # April
        5: 31 * 24 * 60 * 60   # May
    }

    def convert_to_mm_per_month(group):
        month = group.time.dt.month[0].item()  # Get the month number from the first item of the group
        return group * 1000 * seconds_per_month[month]

    # Apply conversion on grouped data
    precip_mm_month = precip_anm.groupby('time.month').map(convert_to_mm_per_month)

    precip_anm = precip_mm_month.resample(time='1Y').mean(dim='time')

    return precip_anm


def draw_correlation_timeseries(data, ax, linestyle='-', linecolor='k', legend_name='label', color='k'):
    """simple timeseries plotting function"""
    ax.plot(
        data.time,
        data.corr,
        linestyle=linestyle,
        color=linecolor,
        label=legend_name,
        linewidth=1,
    ) 


def draw_regression_map(reg, ax):
    """simple regression map plotting function"""
    im = ax.contourf(
                    lon,
                    lat,
                    reg.values,
                    cmap=cmaps.NCV_blu_red,
                    norm=mpl.colors.CenteredNorm(),
                    transform=ccrs.PlateCarree(),
                    extend="both",
                    levels=[-15, -12.5, -10, -7.5, -5, -2.5, -0.5, 0.5, 2.5, 5, 7.5, 10, 12.5, 15]
                )
    ax.coastlines(linewidth=0.5)
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(), draw_labels=False, linewidth=0.1, color="gray", alpha=0.5
    )
    return im
#####

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