"""
SEAM Project Utility Functions
===================================================================

A collection of useful functions for climate data analysis with a
focus on modes of climate variability, teleconnections, monsoon
rainfall.

"""

from typing import Set, Union, Pattern, List
from scipy.signal import detrend
from statsmodels.stats.multitest import multipletests
import xesmf as xe 

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
from scipy import stats
from scipy.stats import linregress 
from src.inputs import *
from src import cesm_utils, precip, nino34
import xskillscore as xs
import os
import regionmask
import cmaps
import cartopy.crs as ccrs 
import matplotlib as mpl 
import itertools
import gcgridobj
import random
random.seed(42)

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


def open_dataset(file_path: str, cesm=False):
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
    if (ds.lon < 0).any():
        ds.coords['lon'] = (ds.coords['lon'] % 360)
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

    if (ds.lon < 0).any():
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


def get_obs_msea_prect_anomaly_timeseries_mam(prect_data_source, detrend_option=False):
    """Load observational prect data and calculate area-averaged timeseries 
    of MAM anomalies in the MSEA region relative to 1951-2015 base period."""

    # Load observational data
    file = prect_gridded_rain_gauge_source_to_file[prect_data_source] 
    ds = open_dataset(file)
    da = ds["precip"].sel(time=slice("1951-01", "2015-12"))

    # Calculate anomaly timeseries
    msea_prect_anomaly_timeseries = (
        precip.get_msea_anomaly_timeseries(
            da,
            detrend=detrend_option,
            base_start='1951-01',
            base_end='2015-12',
            monthly=True,
        ))

    # Select MAM season/months
    msea_prect_anomaly_timeseries_mam = msea_prect_anomaly_timeseries.sel(
        time=msea_prect_anomaly_timeseries.time.dt.season=="MAM"
        )

    return msea_prect_anomaly_timeseries_mam.resample(time="1Y").mean()


def get_obs_nino34_sst_anomaly_timeseries_djf(sst_data_source, detrend_option=False):
    """Load observational sst data and calculate area-averaged timeseries 
    of DJF anomalies in the Niño3.4 region relative to 1951-2015 base period."""

    # Load observational data
    file = sst_reanalysis_source_to_file[sst_data_source] 
    ds = open_dataset(file)
    da = ds["sst"].sel(time=slice("1951-01", "2015-12"))

    # Calculate anomaly timeseries
    nino34_sst_anomaly_timeseries = (
        nino34.get_nino34_anomaly_timeseries(
        da,
        detrend=detrend_option,
        base_start='1951-01',
        base_end='2015-12',
        filtered=True
    ))

    # Select DJF season/months
    nino34_sst_anomaly_timeseries_seasonal = (
        nino34_sst_anomaly_timeseries.resample(time="QS-DEC", label="left")
        .mean(dim='time')
        .sel(time=slice("1951-01", "2015-12"))
    )
    nino34_sst_anomaly_timeseries_djf = (
        nino34_sst_anomaly_timeseries_seasonal.sel(
            time=nino34_sst_anomaly_timeseries_seasonal.time.dt.month.isin([12])
            )
    )
    return nino34_sst_anomaly_timeseries_djf.resample(time="1Y").mean()


def calculate_runcorr_statistics_timeseries(obs_nino34_sst_anomalies_list, obs_msea_prect_anomalies_list, 
                                                window=13):
        """Take as input list of unique observed Niño3.4 sst and MSEA prect anomaly timeseries and 
        return the mean/max/min of across all unique data pairs."""

        unique_data_pairs = itertools.product(obs_nino34_sst_anomalies_list, obs_msea_prect_anomalies_list)
        
        running_correlations = []
        for each in unique_data_pairs:
            running_correlation = get_running_corr(each[0].shift(time=1), each[1], window=window)
            running_correlations.append(running_correlation.corr)
        
        running_correlations_da = xr.concat(running_correlations, dim='data_sources')
        
        runcorr_mean = running_correlations_da.mean(dim='data_sources')
        runcorr_max = running_correlations_da.max(dim='data_sources')
        runcorr_min = running_correlations_da.min(dim='data_sources')

        return runcorr_mean, runcorr_max, runcorr_min


def regress_index_onto_field(field, index):
    
    field_standardized = field / field.std(dim='time')

    cov = xr.cov(field_standardized, index, dim="time")
    var = field_standardized.var(dim="time", skipna=True)

    regression = cov / var

    return regression


def regress_index_list_onto_field_list(field_list, index_list):
    regressions = []
    unique_data_pairs = itertools.product(field_list, index_list)
    for each in unique_data_pairs:
        regression = regress_index_onto_field(each[0], each[1])
        regressions.append(regression)
    regressions_da = xr.concat(regressions, dim="datasets")
    return regressions_da


def regrid_observed_ssts_to_cesm_grid(obs_global_sst_climatology_list, cesm_da):
    """Take as input a list of observed global SSTs and regrid to CESM grid"""
    
    # Define destination grid
    dst_dataset = cesm_da.to_dataset(name='cesm')
    dst_grid = gcgridobj.latlontools.extract_grid(dst_dataset)
    lat = dst_dataset['lat'].values
    lon = dst_dataset['lon'].values

     # Convert all observation times to ERSST times
    # Times differ in format but are otherwise identical
    ersst_da = obs_global_sst_climatology_list[0].sel(time=slice("1951-01", "2015-12"))
    time = ersst_da.time.values

    regridded_data = []

    # For each unique SST dataset, regrid to match CESM2 grid and then append to a list
    for i, (sst_data) in enumerate(obs_global_sst_climatology_list):
        ds = sst_data.sel(time=slice("1951-01", "2015-12"))

        src_grid = gcgridobj.latlontools.extract_grid(sst_data.to_dataset(name='sst'))
        regridder = gcgridobj.regrid.gen_regridder(src_grid, dst_grid)
        da_regridded = regridder(sst_data)
        sst_data_regridded = xr.DataArray(da_regridded, coords={'time': time, 'lat': lat, 
                                        'lon': lon},
                    dims=['time', 'lat', 'lon'])

        regridded_data.append(sst_data_regridded)

    return regridded_data


def get_cesm_msea_prect_anomaly_timeseries_mam(ds, months, detrend_option=False):
    """Calculate MSEA MAM prect from CESM-LENS2 model output.
    Renamed from "get_model_precip_anomalies" """
    precip_ds = ds.sel(time=slice("1900-01", "2100-12"))
    precip_da = precip_ds["PRECT"]
    precip_anm = (
        precip.get_msea_anomaly_timeseries(
            precip_da,
            detrend=detrend_option,
            base_start='1951-01',
            base_end='2015-12',
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

    precip_mm_month = precip_anm.groupby('time.month').map(convert_to_mm_per_month)
    precip_anm = precip_mm_month.resample(time='1Y').mean(dim='time')

    return precip_anm

def get_cesm_nino34_sst_anomaly_timeseries_djf(ds, detrend_option=False):
    """Calculate Niño3.4 DJF SSTs from CESM-LENS2 model output.
    Renamed from "get_model_sst_anomalies". """
    sst_ds = ds.sel(time=slice("1900-01", "2100-12"))
    sst_da = sst_ds["SST"]
    sst_anm_nino34_ersst = nino34.get_nino34_anomaly_timeseries(
        sst_da, detrend=detrend_option, base_start='1951-01', base_end='2015-12', filtered=True
    )
    sst_season = (
        sst_anm_nino34_ersst.resample(time="QS-DEC", label="left")
        .mean(dim='time')
        .sel(time=slice("1900-01", "2100-12"))
    )  # take quarterly means starting Dec 1
    nino34_DJF_ersst = (
        sst_season.sel(time=sst_season.time.dt.month.isin([12]))
        .resample(time="1Y")
        .mean()
    )

    return nino34_DJF_ersst

def get_cmip_nino34_sst_anomaly_timeseries_djf(da_tos):
    sst_anm_nino34_ersst = (
        nino34.get_nino34_anomaly_timeseries(
            da_tos, 
            detrend=False, 
            base_start='1951-01',
            base_end='2015-12', 
            filtered=False
        )
        )
    sst_season = (
        sst_anm_nino34_ersst.resample(time="QS-DEC", label="left")
            .mean(dim='time')
        )  # take quarterly means starting Dec 1

    nino34_DJF_ersst = (
        sst_season.sel(time=sst_season.time.dt.month.isin([12]))
        .resample(time="1Y")
        .mean()
        )

    return nino34_DJF_ersst.sel(time=slice('1900','2014'))


def get_cmip_msea_prect_anomaly_timeseries_mam(da_pr):
    precip_anm = (
        precip.get_msea_anomaly_timeseries(
            da_pr,
            detrend=False,
            base_start='1951-01',
            base_end='2015-12',
            monthly=True)
        )

    precip_MAM = precip_anm.sel(time=precip_anm.time.dt.season=="MAM")
    precip_anm = precip_MAM.resample(time="1Y").mean()

    return precip_anm.sel(time=slice('1900','2014'))


def get_running_corr(array1, array2, window=13, min_periods=5, center=True):
    """apply a rolling correlation coefficient"""
    s1 = pd.Series(array1)
    s2 = pd.Series(array2)
    corr = s1.rolling(window, min_periods=min_periods, center=center).corr(s2)

    ds = xr.Dataset(
        data_vars=dict(
            corr=(["time"], corr.values),
        ),
        coords=dict(
            time=array1.time,
        ),
    )

    return ds


def select_random_timeseries(sst_data, prect_data, num_samples=10000):
    """Randomly select 10,000 13-year timeseries. 
    For each 13-year period, calculate correlation 
    and store ensemble member idx and start-year."""

    start_year = sst_data.time.dt.year.values[0]
    end_year = sst_data.time.dt.year.values[-1]+1
    years = list(range(start_year, end_year))

    
    correlations = []
    random_members = []
    random_starts = []

    for _ in range(num_samples):

        # Generate a random time index (0=1900, 102=2014-1900-13+1)
        random_time = random.randint(start_year-start_year, end_year - start_year - 13)

        # Generate a random ensemble member index (between 0 and 100)
        random_member = random.randint(0, 99)

        pre = prect_data.isel(ensemble=random_member).isel(time=slice(random_time,random_time+13))
        st = sst_data.isel(ensemble=random_member).isel(time=slice(random_time,random_time+13))

        corr = xr.corr(pre, st.shift(time=1), dim='time')

        correlations.append(corr.values.item())
        random_members.append(random_member)
        random_starts.append(years[random_time])

    df = pd.DataFrame({'Correlations': correlations, 'Members': random_members, 'Years': random_starts})

    return df


def select_correlation_quartiles(df, sort_by="Correlations"):
    """Take as input any dataframe and return the
    upper quartile, lower quartile, and a random quartile
    of same size."""

    df_sorted = df.sort_values(by=[sort_by])

    q1 = df_sorted[sort_by].quantile(0.25)
    q3 = df_sorted[sort_by].quantile(0.75)

    # Filter rows in the top quartile
    top_quartile = df_sorted[df_sorted['Correlations'] >= q3]
    top_quartile = top_quartile.dropna()
    upper_df = top_quartile

    # Filter rows in the bottom quartile
    bottom_quartile = df_sorted[df_sorted['Correlations'] <= q1]
    bottom_quartile = bottom_quartile.dropna()
    lower_df = bottom_quartile

    # Create random quartile
    if upper_df.shape[0]==lower_df.shape[0]:
        random_df = df_sorted.sample(int(upper_df.shape[0]), random_state=42)
    else:
        random_df_len_upper = df_sorted.sample(int(upper_df.shape[0]), random_state=42)
        random_df_len_lower = df_sorted.sample(int(lower_df.shape[0]), random_state=42)

    return upper_df, lower_df, random_df_len_upper, random_df_len_lower


def select_field_quartiles(field_da, dataframe_categories):
    """
    Select global field slices for the given categories.
    Categories are dataframes that contain a list of ensemble members ("Members")
    and a list of start years for the 13-year periods("Years")
    
    Parameters:
        field_da (xarray.DataArray): the DataArray containing fields.
        dataframe_categories (list of pd.DataFrame): List containing dataframes for
                                                     each category.
    
    Returns:
        tuple: three xr.DataArrays containing field slices for the given categories.
    """
    
    def get_slices(df):
        """Helper function to extract field slices for a given dataframe."""
        slices = []
        for _, row in df.iterrows():
            ens = row['Members']
            year = row['Years']

            field_ens = field_da.isel(ensemble=int(ens))
            my_slice = field_ens.sel(
                time=slice(str(int(year)), str(int(year) + 13))
                ).mean(dim='time')
            slices.append(my_slice)
        return slices

    # Apply the helper function to each category dataframe
    category_slices = [xr.concat(get_slices(df), dim='random') for df in dataframe_categories]

    return category_slices



def correct_pvals(p_val, alpha_global, method='fdr=bh'):
    """Apply the Benjamini-Hochberg correction to correct p-values,
    accounting for multiple hypothesis testing."""
    
    p_values = p_val

    # Flatten array and remove NaNs
    p_values_flat = p_values.flatten()
    non_nan_mask = ~np.isnan(p_values_flat)
    non_nan_pvals = p_values_flat[non_nan_mask]

    # Apply the Benjamini-Hochberg correction
    _, corrected_pvals, _, _ = multipletests(non_nan_pvals, alpha=alpha_global, method=method)

    # Place corrected p-values back into the original array, keeping NaNs intact
    corrected_p_values = np.full_like(p_values_flat, np.nan)
    corrected_p_values[non_nan_mask] = corrected_pvals

    # Reshape back to the original shape
    corrected_p_values = corrected_p_values.reshape(p_values.shape)

    return corrected_p_values




def get_cmip6_da(file, ds_out):
    """Function to open CMIP6 data files from a particular model and
    regrid to common ds_out grid with same dimension conventions."""

    da = xr.open_mfdataset(file, engine='netcdf4')

    if da.parent_source_id == "CIESM":
        if "pr" in da.variables:
            da = da*1e3 # fix bug in CIESM pr data that needs to be multiplied by 1000

    if "lat" in da.coords and "lon" in da.coords:
        return da
    elif "latitude" in da.coords and "longitude" in da.coords:
        da = da.rename({"latitude": "lat", "longitude": "lon"})
    elif "nav_lat" in da.coords and "nav_lon" in da.coords:
        da = da.rename({"nav_lat": "lat", "nav_lon": "lon"})

    if (da.lon < 0).any():
        da.coords['lon'] = (da.coords['lon'] % 360)

    if "bnds" in da.coords:  # and "lon" in da.coords:
        da = da.drop_dims(["bnds"])
    if "vertices" in da.coords:
        da = da.drop_dims(["vertices"])
    if "nvertex" in da.coords:
        da = da.drop_dims(["nvertex"])
    if "axis_nbounds" in da.coords:
        da = da.drop_dims(["axis_nbounds"])

    # Using nearest_s2d for: IPSL-CM6A-LR, MIROC6, and a few others
    # Otherwise bilinear
    if "i" in da.coords:  # and "lon" in da.coords:
        regridder = xe.Regridder(da, ds_out, 'nearest_s2d')
    elif "x" in da.dims:  # and "lon" in da.coords:
        regridder = xe.Regridder(da, ds_out, 'nearest_s2d')
    else:
        regridder = xe.Regridder(da, ds_out, 'bilinear')
    
    da_new = regridder(da)

    return da_new












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



#####



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
    if (ds.lon < 0).any():
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
    if (ds.lon < 0).any():
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
    if (ds.lon < 0).any():
        ds.coords['lon'] = (ds.coords['lon'] % 360)
        #TODO: Check if I need to reindex?
        ds = ds.sortby(ds.lon)
        
    return ds


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



def mask_ocean(da):
    """Mask ocean areas"""
    land = regionmask.defined_regions.natural_earth_v5_0_0.land_110
    mask = land.mask(da["lon"], da["lat"])
    da_new = da.where(~np.isnan(mask))
    return da_new
    

def mask_land(da):
    """Mask land areas"""
    land = regionmask.defined_regions.natural_earth_v5_0_0.land_110
    mask = land.mask(da["lon"], da["lat"])
    da_new = da.where(np.isnan(mask))
    return da_new



def ttest_1samp(a, popmean, dim):
    """
    TO DO: add documentation
    """
    t_val, p_val = stats.ttest_1samp(a, popmean, axis=a.get_axis_num(dim))

    return t_val, p_val





