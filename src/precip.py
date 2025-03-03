"""
precip.py
===================================================================

Functions for subsetting precipitation output in MSEA region.

"""

import xarray as xr
from src import utils

def get_SEAM_map(precip_da: xr.DataArray, monthly=False, detrend=False,
                     base_start=None, base_end=None,
                     anomaly=False):
    """Get a large map of precipitation climatology (or anomalies) in the greater
    continental Southeast Asia region (larger than study area).

    Note: Anomalies are calculated relative to the entire climatological
    period

    Args:
        precip_da (xr.DataArray): precipitation data (i.e. GPCC)
        detrend (bool) : if True, removes the linear trend

    Optional Args:
        monthly (bool, default: False): if True, return monthly
        precipitation anomalies (i.e. for regression or corrrelation
        with other monthly values), otherwise return annual means

        monsoon_season (bool, default: True): if True, select only
        May - October months
    """
    utils.check_data_conventions(precip_da)

    # Select region around SEA (80, 120, 0, 35)
    maxlat = 35
    minlat = 0
    minlon = 80
    maxlon = 120

    cropped_da = precip_da.sel(
        lat=slice(minlat, maxlat), lon=slice(minlon, maxlon)
    )

    # Calculate GPCC anomalies relative to climatology for entire period
    if anomaly:
        anm = utils.remove_monthly_clm(cropped_da, base_start, base_end)
    else:
        anm = cropped_da

    # Detrend (optional)
    if detrend:
        anm = utils.detrend_array(anm, dim="time")

    # Calculate annual anomalies (optional)
    if monthly is False:
        anm = anm.resample(time='1Y').mean()

    return anm


def get_msea_anomaly_timeseries(precip_da: xr.DataArray, detrend: bool,
                            base_start: str, base_end: str,
                            monthly=False):
    """Get a timeseries of precipitation anomalies averaged over
       the continental Southeast Asia region.

    Note: base start and end month must always be specified

    Args:
        precip_da (xr.DataArray): precipitation data (i.e. GPCC)
        detrend (bool): if True, remove the linear trend
        base_start (str)  : start year and month of base period (i.e. '1951-01')
        base_end (str)    : end year and month of base period (i.e. '2015-12')

    Optional Args:
        monthly (bool, default: False): if True, return monthly precipitation
        anomalies, otherwise aggregate into annual values

        monsoon_season (bool, default: True): if True, select only
        May-October months
    """
    utils.check_data_conventions(precip_da)
    
    # Select MSEA region (90, 110, 10, 25)
    maxlat = 25
    minlat = 10
    minlon = 90
    maxlon = 110

    da = precip_da.sel(
        lat=slice(minlat, maxlat), lon=slice(minlon, maxlon)
    ).compute()

    # Detrend (optional)
    if detrend:
        da = utils.detrend_array(da, dim="time")

    # Calculate prect anomalies by removing monthly climatology
    da = utils.remove_monthly_clm(da, base_start, base_end)

    # Calculate the area averaged mean
    da = utils.calc_cos_wmean(da)

    # Calculate annual anomalies (optional)
    if monthly is False:
        da = da.resample(time='1Y').mean()

    return da


def get_msea_climatology_timeseries(precip_da: xr.DataArray, detrend: bool,
                            monthly=False):
    """Get a timeseries of precipitation anomalies averaged over
       the continental Southeast Asia region.

    Note: base start and end month must always be specified

    Args:
        precip_da (xr.DataArray): precipitation data (i.e. GPCC)
        detrend (bool): if True, remove the linear trend
        base_start (str)  : start year and month of base period (i.e. '1951-01')
        base_end (str)    : end year and month of base period (i.e. '2015-12')

    Optional Args:
        monthly (bool, default: False): if True, return monthly precipitation
        anomalies, otherwise aggregate into annual values

        monsoon_season (bool, default: True): if True, select only
        May-October months
    """
    utils.check_data_conventions(precip_da)
    
    # Select MSEA region (90, 110, 10, 25)
    maxlat = 25
    minlat = 10
    minlon = 90
    maxlon = 110

    da = precip_da.sel(
        lat=slice(minlat, maxlat), lon=slice(minlon, maxlon)
    ).compute()

    # Detrend (optional)
    if detrend:
        da = utils.detrend_array(da, dim="time")

    # Calculate the area averaged mean
    da = utils.calc_cos_wmean(da)

    # Calculate annual anomalies (optional)
    if monthly is False:
        da = da.resample(time='1Y').mean()

    return da


def get_SEAM_clm_timeseries(precip_da: xr.DataArray, detrend: bool,
                            monsoon_season=True, monthly=False):
    """Get a timeseries of precipitation anomalies averaged over
       the continental Southeast Asia region.

    Args:
        precip_da (xr.DataArray): precipitation data (i.e. GPCC)
        detrend (bool): if True, remove the linear trend
        base_start (str)  : start year and month of base period (i.e. '1951-01')
        base_end (str)    : end year and month of base period (i.e. '2015-12')

    Optional Args:
        monthly (bool, default: False): if True, return monthly precipitation
        climatology, otherwise aggregate into annual values

        monsoon_season (bool, default: True): if True, select only
        May-October months
    """
    utils.check_data_conventions(precip_da)
    
    # Select MSEA region (90, 110, 10, 25)
    maxlat = 25
    minlat = 10
    minlon = 90
    maxlon = 110

    da = precip_da.sel(
        lat=slice(minlat, maxlat), lon=slice(minlon, maxlon)
    )

    # Select only monsoon season months (optional)
    if monsoon_season:
        da = da.sel(
            time=da.time.dt.month.isin([5, 6, 7, 8, 9, 10])
        )

    # Detrend (optional)
    if detrend:
        da = utils.detrend_array(da, dim="time")

    # Calculate the area averaged mean
    da = utils.calc_cos_wmean(da)

    # Calculate annual anomalies (optional)
    if monthly is False:
        da = da.resample(time='1Y').mean()

    return da