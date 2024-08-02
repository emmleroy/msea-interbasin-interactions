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

from seam import utils


def get_IOD_west_area(sst_da: xr.DataArray):
    """Return the area slices for the IOB region
       IOD west: 50°E to 70°E and 10°S to 10°N.

    Args:
        sst_da (xr.DataArray) : SST data (i.e. HadISST, HadSST4, ERSST)

    Returns:
        IOB_region (xr.DataArray): SST data sliced to the Nino3.4 region

    """

    utils.check_data_conventions(sst_da)

    IOD_west = sst_da.sel(lat=slice(-10, 10), lon=slice(50, 70))

    return IOD_west

def get_IOD_east_area(sst_da: xr.DataArray):
    """Return the area slices for the IOB region
       IOD east: 90°E to 110°E and 10°S to 0°S

    Args:
        sst_da (xr.DataArray) : SST data (i.e. HadISST, HadSST4, ERSST)

    Returns:
        IOB_region (xr.DataArray): SST data sliced to the Nino3.4 region

    """

    utils.check_data_conventions(sst_da)

    IOD_east = sst_da.sel(lat=slice(-10, 0), lon=slice(90, 110))

    return IOD_east


def get_IOD_anm_timeseries(sst_da: xr.DataArray, detrend: bool,
                          base_start: str, base_end: str,
                          filtered=True):
    """

    """

    # Default filter: 3 month rolling mean
    if filtered:
        sst_da = sst_da.rolling(time=3, center=True, min_periods=1).mean()

    # Detrend the data (optional)
    if detrend:
        sst_da = utils.detrend_array(sst_da, "time", 1)

    # Calculate SST anomalies by removing monthly climatology
    # (default climatological period is the entire data period)
    sst_anm = utils.remove_monthly_clm(sst_da, base_start, base_end)

    EAST = get_IOD_east_area(sst_anm).load()
    WEST = get_IOD_west_area(sst_anm).load()

    EAST_mean = utils.calc_cos_wmean(EAST)
    WEST_mean = utils.calc_cos_wmean(WEST)

    diff = WEST_mean-EAST_mean

    return diff/diff.std(dim='time')
