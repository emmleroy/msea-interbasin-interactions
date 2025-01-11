"""
nino34.py
===================================================================

These functions calculate the Nino3.4 Index for the El Nino -
Southern Oscillation (ENSO) based on monthly global SST data. 

"""

import xarray as xr
import numpy as np
import pandas as pd
from datetime import timedelta
import copy
from scipy import signal

from src import utils


def get_nino34_area(sst_da: xr.DataArray):
    """Return the area slices for the Nino3.4 region
       (5°N-5°S, 170°W-120°W).
       Note: 170°W = 190°E and 120°W = 240°E

    Args:
        sst_da (xr.DataArray) : SST data (i.e. HadISST, HadSST4, ERSST)

    Returns:
        nino34_region (xr.DataArray): SST data sliced to the Nino3.4 region

    """

    utils.check_data_conventions(sst_da)

    nino34_region = sst_da.sel(lat=slice(-5, 5), lon=slice(190, 240))

    return nino34_region.compute()


def get_nino3_area(sst_da: xr.DataArray):
    """Return the area slices for the Nino3.4 region
       (5°N-5°S, 90°W-150°W).

    Args:
        sst_da (xr.DataArray) : SST data (i.e. HadISST, HadSST4, ERSST)

    Returns:
        nino34_region (xr.DataArray): SST data sliced to the Nino3.4 region

    """

    utils.check_data_conventions(sst_da)

    nino3_region = sst_da.sel(lat=slice(-5, 5), lon=slice(210, 270))

    return nino3_region


def get_nino4_area(sst_da: xr.DataArray):
    """Return the area slices for the Nino4 region
       (5°N-5°S,  160E-150W).

    Args:
        sst_da (xr.DataArray) : SST data (i.e. HadISST, HadSST4, ERSST)

    Returns:
        nino34_region (xr.DataArray): SST data sliced to the Nino3.4 region

    """

    utils.check_data_conventions(sst_da)

    nino4_region = sst_da.sel(lat=slice(-5, 5), lon=slice(160, 210))

    return nino4_region


def get_nino34_anomaly_timeseries(sst_da: xr.DataArray, detrend: bool,
                        base_start: str, base_end: str,
                          filtered=True):
    """Calculate SST anomalies in the Nino3.4 region relative to climatology.

    Note that the climatological period is the entire time period spanned by
    the input sst data. Must specify option to remove the linear trend.
    Default filtering is a 3-month rolling mean.

    Args:
        sst_da (xr.DataArray) : SST data (i.e. HadISST, HadSST4, ERSST)

        detrend (bool): if True, remove the linear trend

        filtered (bool, default: True): if True, apply a rolling 3-month mean

    Returns:
        nino34_sst_anm (xr.DataArray): SST anomalies in the Nino3.4 region in
        units of temperature.

    """
    utils.check_data_conventions(sst_da)

    # Select Niño3.4 region
    sst_slice = get_nino34_area(sst_da)

    # Detrend the data (optional)
    if detrend:
        # Detrend by removing timeseries of global mean SSTs
        #global_mean_ssts = utils.calc_cos_wmean(sst_da)
        #sst_slice = sst_slice-global_mean_ssts
        # Detrend by removing linear trend (does not work for IOB)
        sst_slice = utils.detrend_array(sst_slice, "time", 1)

    # Default filter: 3 month rolling mean
    if filtered:
        sst_slice = sst_slice.rolling(time=3, center=True, min_periods=1).mean()

    # Calculate SST anomalies by removing monthly climatology
    sst_anm = utils.remove_monthly_clm(sst_slice, base_start, base_end)

    # Calculate area-weighted mean of SST anomalies
    nino34_sst_anm = utils.calc_cos_wmean(sst_anm)

    return nino34_sst_anm.squeeze()


def get_nino4_anm_timeseries(sst_da: xr.DataArray, detrend: bool,
                        base_start: str, base_end: str,
                          filtered=True, decadal=False):
    """Calculate SST anomalies in the Nino4 region relative to climatology.

    Note that the climatological period is the entire time period spanned by
    the input sst data. Must specify option to remove the linear trend.
    Default filtering is a 3-month rolling mean.

    Args:
        sst_da (xr.DataArray) : SST data (i.e. HadISST, HadSST4, ERSST)

        detrend (bool): if True, remove the linear trend

        filtered (bool, default: True): if True, apply a rolling 3-month mean

    Returns:
        nino4_sst_anm (xr.DataArray): SST anomalies in the Nino4 region in
        units of temperature.

    """

    # Get slice of SST data for the Nino4 region
    sst_slice = get_nino4_area(sst_da)

    # Detrend the data (optional)
    if detrend:

        # Detrend by removing timeseries of global mean SSTs
        global_mean_ssts = utils.calc_cos_wmean(sst_da)
        sst_slice = sst_slice-global_mean_ssts

        # Detrend by removing linear trend (does not work for IOB)
         #sst_slice = utils.detrend_array(sst_slice, "time", 1) 

    # Default filter: 3 month rolling mean
    if filtered:
        sst_slice = sst_slice.rolling(time=3, center=True, min_periods=1).mean()

    # Calculate SST anomalies by removing monthly climatology
    # (default climatological period is the entire data period)
    sst_anm = utils.remove_monthly_clm(sst_slice, base_start, base_end)

    # Calculate area-weighted mean of SST anomalies
    nino4_sst_anm = utils.calc_cos_wmean(sst_anm)

    # Apply 13-year Chebyshev filter (optional)
    if decadal:
        nino4_sst_anm = _apply_cheby1_filter(nino4_sst_anm)

    return nino4_sst_anm.compute()


def _apply_cheby1_filter(da: xr.DataArray, period=13*12, btype='lowpass', n=6.0,
                        rp=0.1, fs=None):
    """Apply a Chebyshev type I digital filter to monthly timeseries data.

    Default filter design values are from Henley et al. 2015 (i.e. 13-year
    low-pass filter with period 6 for the TPI Index).

    This function was originally copied from Sonya Wellby's github repo:
    https://github.com/sonyawellby/anu_honours/blob/master/tpi.py.

    Args:
        da (xr.DataArray): monthly timeseries data to be filtered
        period (float, default: 13*12): period of the filter in months
        n (float, default: 6): the order of the filter,
        rp (float, default: 0.1): the peak to peak passband ripple (in decibles)
        fs (bool, default: False): the sampling frequency of the system
                             default = False

    Returns:
        da_filt (xr.DataArray): filtered monthly timeseries data

    """

    wn = 1/(period*0.5)    # critical frequencies --> half-cycles / sample
    b, a = signal.cheby1(n, rp, wn, btype=btype, analog=False,
                         output='ba', fs=fs)
    nda_filt = signal.filtfilt(b, a, da)  # output is a filtered numpy.ndarray

    # Convert numpy.ndarray 'nda_filt' to an xr.DataArray
    series = pd.Series(data=nda_filt, index=da.time.values)
    da_filt = xr.DataArray.from_series(series)
    da_filt = da_filt.rename({'index': 'time'})

    return da_filt


def get_nino4_clm_timeseries(sst_da: xr.DataArray, detrend: bool,
                          filtered=True):
    """Calculate SST climatology in the Nino3.4 region relative to climatology.

    Note that the climatological period must now be specific. Must also specify
    option to remove the linear trend.
    Default filtering is a 3-month rolling mean.

    Args:
        sst_da (xr.DataArray) : SST data (i.e. HadISST, HadSST4, ERSST)

        detrend (bool): if True, remove the linear trend

        filtered (bool, default: True): if True, apply a rolling 3-month mean

    Returns:
        nino4_sst_clm (xr.DataArray): SST climatology in the Nino3.4 region in
        units of temperature.

    """

    # Get slice of SST data for the Nino3.4 region
    sst_slice = get_nino4_area(sst_da)

    # Default filter: 3 month rolling mean
    if filtered:
        sst_slice = sst_slice.rolling(time=3, center=True, min_periods=1).mean()

    # Detrend the data (optional)
    if detrend:
        sst_slice = utils.detrend_array(sst_slice, "time", 1)

    # Calculate area-weighted mean of SST anomalies
    nino4_sst_clm = utils.calc_cos_wmean(sst_slice)

    return nino4_sst_clm


def get_nino34_clm_timeseries(sst_da: xr.DataArray, detrend: bool,
                          filtered=True):
    """Calculate SST climatology in the Nino3.4 region relative to climatology.

    Note that the climatological period must now be specific. Must also specify
    option to remove the linear trend.
    Default filtering is a 3-month rolling mean.

    Args:
        sst_da (xr.DataArray) : SST data (i.e. HadISST, HadSST4, ERSST)

        detrend (bool): if True, remove the linear trend

        filtered (bool, default: True): if True, apply a rolling 3-month mean

    Returns:
        nino34_sst_clm (xr.DataArray): SST climatology in the Nino3.4 region in
        units of temperature.

    """

    # Get slice of SST data for the Nino3.4 region
    sst_slice = get_nino34_area(sst_da)

    # Default filter: 3 month rolling mean
    if filtered:
        sst_slice = sst_slice.rolling(time=3, center=True, min_periods=1).mean()

    # Detrend the data (optional)
    if detrend:
        sst_slice = utils.detrend_array(sst_slice, "time", 1)

    # Calculate area-weighted mean of SST anomalies
    nino34_sst_clm = utils.calc_cos_wmean(sst_slice)

    return nino34_sst_clm