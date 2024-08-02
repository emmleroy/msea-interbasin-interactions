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


def get_nino34_anm_timeseries(sst_da: xr.DataArray, detrend: bool,
                        base_start: str, base_end: str,
                          filtered=True, decadal=False):
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

    # Get slice of SST data for the Nino3.4 region
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
    # (default climatological period is the entire data period)
    sst_anm = utils.remove_monthly_clm(sst_slice, base_start, base_end)

    # Calculate area-weighted mean of SST anomalies
    nino34_sst_anm = utils.calc_cos_wmean(sst_anm)

    # Apply 13-year Chebyshev filter (optional)
    if decadal:
        nino34_sst_anm = _apply_cheby1_filter(nino34_sst_anm)

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


def get_ENSO_years(sst_da, base_start: str, base_end: str):
    mean = get_nino34_anm_timeseries(sst_da, detrend=True, 
                                    base_start=base_start, base_end=base_start,
                                    filtered=True)
    std = mean.std(dim='time')

    norm = mean/std  
    norm_seasonal = (
        norm.resample(time="QS-DEC", label="left")
        .mean()
    )

    norm_DJF = (
        norm_seasonal.sel(time=norm_seasonal.time.dt.month.isin([12]))
        .resample(time="1Y")
        .mean()
    )

    df = norm_DJF.to_dataframe(name="DJF Nino3.4")

    El_Nino_Years = df.where(df>0.75).dropna().index.year
    La_Nina_Years = df.where(df<0.75).dropna().index.year

    return El_Nino_Years.tolist(), La_Nina_Years.tolist()



def get_ENSO_years2(sst_da, with_months=False, start_only=True):

    anm = get_nino34_anm_timeseries(sst_da, detrend=True, filtered=True)

    # Define thresholds
    upper_threshold = np.percentile(anm.dropna(dim='time').values, 75)
    lower_threshold = np.percentile(anm.dropna(dim='time').values, 25)

    print("upper threshold = ", upper_threshold)
    print("lower threshold = ", lower_threshold)

    # Create boolean arrays where smoothed Nino3.4 index exceeds threshold
    nino_bool = xr.where(
        anm >= upper_threshold, 1, 0  # 1 if condition is met
    )
    nina_bool = xr.where(
        anm <= lower_threshold, 1, 0  # 0 otherwise
    )

    # Calculate a running 5-month sum of the boolean arrays
    nino_bool_sum = nino_bool.rolling(time=5, min_periods=5, center=True).sum()
    nina_bool_sum = nina_bool.rolling(time=5, min_periods=5, center=True).sum()

    # Select nino/nina periods when the smoothed SST anomalies exceed the threshold
    # for at least 5 months
    nino_periods = (
        nino_bool_sum.where(nino_bool_sum == 5).dropna(dim="time").time
    )
    nina_periods = (
        nina_bool_sum.where(nina_bool_sum == 5).dropna(dim="time").time
    )

    # Get unique years that include DJF of previous year
    nino_years = np.unique((nino_periods.time.dt.year))
    nina_years = np.unique((nina_periods.time.dt.year))

    all_nino_years = [
        1900,1902,1904,1905,1911,1912, 
        1913,1914,
        1918,1919,1923,1925, 1926,
        1930,1931, 1940,1941,1951,1953,
        1957,1958, 1963,1965,1968,
        1972,1976,1977,1982,1983,
        1986,1987,1991,1992,1994,1997,2002,
        2004,2006,2009,2015,2018,2019
    ]

    all_nina_years = [
        1903, 1908, 1909, 1916,
        1921, 1922, 1924, 1933,
        1938, 1942, 1944, 1948,
        1949, 1950, 1954, 1956,
        1964, 1970, 1973, 1974, 1975,
        1984, 1985, 1988, 1989, 1995, 1998,
        2000, 
        2007, 2008, 2010, 2011, 2017
    ]

    my_nino_starts = np.array(
        ["1900-02",
        "1902-03",
        "1904-10",
        "1911-07",
        "1913-11",
        "1918-08",
        "1919-11",
        "1923-09",
        "1925-09",
        "1930-06",
        "1940-01",
        "1951-07",
        "1953-02",
        "1957-04",
        "1963-07",
        "1965-06",
        "1968-11",
        "1972-05",
        "1976-09",
        "1977-09",
        "1982-05",
        "1986-09",
        "1991-06",
        "1994-07",
        "1997-05",
        "2002-06",
        "2004-07",
        "2006-09",
        "2009-07",
        "2015-03",
        "2018-09"],
        dtype="datetime64",
    )

    df1 = pd.DataFrame(my_nino_starts, dtype="datetime64[ns]")
    nino_starts = df1[0].to_xarray()

    my_nina_starts = np.array(
        ["1903-08",
        "1908-10",
        "1909-08",
        "1916-07",
        "1921-01",
        "1922-07",
        "1924-07",
        "1933-04",
        "1938-01",
        "1942-07",
        "1944-10",
        "1948-07",
        "1949-09",
        "1950-10",
        "1954-05",
        "1956-07",
        "1964-05",
        "1970-07",
        "1973-05",
        "1975-03",
        "1984-11",
        "1988-05",
        "1995-09",
        "1998-07",
        "2007-08",
        "2010-07",
        "2011-09",
        "2017-10"],
        dtype="datetime64",
    )

    df2 = pd.DataFrame(my_nina_starts, dtype="datetime64[ns]")
    nina_starts = df2[0].to_xarray()

    nino_s = xr.Dataset({"dates": nino_starts})
    nina_s = xr.Dataset({"dates": nina_starts})

    nino_s = nino_s.assign_coords(coords={"dates": nino_starts})
    nina_s = nina_s.assign_coords(coords={"dates": nina_starts})

    if with_months is not False:
        return nino_s['dates'], nina_s['dates']
    elif start_only is not False:
        return nino_s['dates'].dt.year, nina_s['dates'].dt.year
    else:
        return all_nino_years, all_nina_years
