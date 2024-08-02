"""
Nino3.4 Index Module
===================================================================

These functions calculate the Nino3.4 Index for the El Nino -
Southern Oscillation (ENSO) based on monthly global SST data.

This module is still in development as I test different ENSO indices
in get_ENSO_years().

Usage:
    Main function to call is :
    nino34_ts = get_nino34_timeseries(sst_da, detrend=True)


"""

# Last updated 4 August 2022 by Emmie Le Roy

import xarray as xr
import numpy as np
import pandas as pd
from datetime import timedelta
import copy
from scipy import signal
from seam import utils


def get_IOB_area(sst_da: xr.DataArray):
    """Return the area slices for the IOB region
       (40–100°E, 30°S–30°N).

    Args:
        sst_da (xr.DataArray) : SST data (i.e. HadISST, HadSST4, ERSST)

    Returns:
        IOB_region (xr.DataArray): SST data sliced to the Nino3.4 region

    """

    utils.check_data_conventions(sst_da)

    IOB_region = sst_da.sel(lat=slice(-30, 30), lon=slice(40, 100))

    return IOB_region

def get_wIOB_area(sst_da: xr.DataArray):
    """Return the area slices for the IOB region
       (40–100°E, 30°S–30°N).

    Args:
        sst_da (xr.DataArray) : SST data (i.e. HadISST, HadSST4, ERSST)

    Returns:
        IOB_region (xr.DataArray): SST data sliced to the Nino3.4 region

    """

    utils.check_data_conventions(sst_da)

    IOB_region = sst_da.sel(lat=slice(-30, 30), lon=slice(40, 70))

    return IOB_region


def _apply_cheby1_filter(da: xr.DataArray, period=13*12, btype='lowpass', n=6.0,
                        rp=0.1, fs=None):
    """Apply a Chebyshev type I digital filter to monthly timeseries data.

    Default filter design values are from Henley et al. 2015 (i.e. 13-year
    low-pass filter with period 6 for the TPI Index).

    This function was originally copied from Sonya Wellby's github repo:
    https://github.com/sonyawellby/anu_honours/blob/master/tpi.py
    but I think her comments on the default filter parameters are wrong
    (rp is not 13 and wn is not 0.1).

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


def get_IOB_anm_timeseries(sst_da: xr.DataArray, detrend: bool,
                          base_start: str, base_end: str,
                          filtered=True, decadal=False):
    """

    """

    # Get IOB area
    IOB = get_IOB_area(sst_da)

    # Detrend the data (optional)
    if detrend:

        # Detrend by removing timeseries of global mean SSTs
        global_mean_ssts = utils.calc_cos_wmean(sst_da)
        IOB = IOB-global_mean_ssts

        # Detrend by removing linear trend (does not work for IOB)
        #IOB = utils.detrend_array(IOB, "time", 1) 

    # Default filter: 3 month rolling mean
    if filtered:
        IOB = IOB.rolling(time=3, center=True, min_periods=1).mean()

    # Calculate SST anomalies by removing monthly climatology
    # (default climatological period is the entire data period)
    IOB_anm = utils.remove_monthly_clm(IOB, base_start, base_end)

    IOB_mean = utils.calc_cos_wmean(IOB_anm)

    # Apply 13-year Chebyshev filter (optional)
    if decadal:
        IOB_mean = _apply_cheby1_filter(IOB_mean)

    return IOB_mean


def get_IOB_clm_timeseries(sst_da: xr.DataArray, detrend: bool,
                          filtered=True, decadal=False):
    """

    """

    # Default filter: 3 month rolling mean
    if filtered:
        sst_da = sst_da.rolling(time=3, center=True, min_periods=1).mean()

    # Detrend the data (optional)
    if detrend:
        sst_da = utils.detrend_array(sst_da, "time", 1)

    IOB = get_IOB_area(sst_da).load()

    IOB_mean = utils.calc_cos_wmean(IOB)

        # Apply 13-year Chebyshev filter (optional)
    if decadal is not False:
        IOB_mean = _apply_cheby1_filter(IOB_mean)

    return IOB_mean

def get_wIOB_clm_timeseries(sst_da: xr.DataArray, detrend: bool,
                          filtered=True, decadal=False):
    """

    """

    # Default filter: 3 month rolling mean
    if filtered:
        sst_da = sst_da.rolling(time=3, center=True, min_periods=1).mean()

    # Detrend the data (optional)
    if detrend:
        sst_da = utils.detrend_array(sst_da, "time", 1)

    IOB = get_wIOB_area(sst_da).load()

    IOB_mean = utils.calc_cos_wmean(IOB)

        # Apply 13-year Chebyshev filter (optional)
    if decadal is not False:
        IOB_mean = _apply_cheby1_filter(IOB_mean)

    return IOB_mean