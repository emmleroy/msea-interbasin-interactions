# 
"""
Functions to calculate tropical interbasin interaction index (TTP)
"""

import numpy as np
import pandas as pd
import xarray as xr
from scipy import signal

from src import utils

def _get_interbasin_areas_modified(da: xr.DataArray):
    """IO = wIOD area, IP = eIOD area, WP = nino3 area 

    """

    utils.check_data_conventions(da)

    # wIOD region
    area1 = da.sel(lat=slice(-10, 10), lon=slice(50, 70))

    # original MC region
    area2 = da.sel(lat=slice(-20, 20), lon=slice(100, 125))

    # Nino4 region
    area3 = da.sel(lat=slice(-5, 5), lon=slice(160, 210)) #160°E–150°W

    return area1, area2, area3


def _apply_interbasin_formula(SSTA_area1, SSTA_area2, SSTA_area3):
    """Calculate the unfiltered interbasin index from SST anomalies in each of
    the three interbasin regions using the interbasin formula.

    Args:

        # tropial west Indian Ocean: 40–80°E, -5°S–5°N)
        SSTA_area1 (xr.DataArray): SSTAs in region 1 (5°S-5°N, 40°E-80°E)
        
        # Central: 100–125°E, -20°S–20°N)
        SSTA_area2 (xr.DataArray): SSTAs in region 2 (20°S-20°N, 100°E-125°E)
        
        #Western Pacific: 5S-5N and 140E-180W
        SSTA_area3 (xr.DataArray): SSTAs in region 3 (5°S-5°N, 140°E-180°E)

    Returns:
        TPI_idx: SSTA_area2 - ((SSTA_area1 + SST_Aarea3)/2)

    """

    a1a3 = SSTA_area1+SSTA_area3
    div2 = a1a3/2.0
    unfilt_interbasin_idx = div2-SSTA_area2

    return unfilt_interbasin_idx


def _apply_cheby1_filter(da: xr.DataArray, period=13*1, btype='lowpass', n=6.0,
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
    
    # Convert numpy.ndarray 'nda_filt' to an xr.DataArray
    if 'ensemble' in da.dims:
        nda_filt = signal.filtfilt(b, a, da.values, axis=1)  # output is a filtered numpy.ndarray
        da_filt = xr.DataArray(nda_filt, dims=('ensemble', 'time'), coords={'time': da.coords['time']})
    else:
        nda_filt = signal.filtfilt(b, a, da.values, axis=0)  # output is a filtered numpy.ndarray
        da_filt = xr.DataArray(nda_filt, dims=('time'), coords={'time': da.coords['time']})

    return da_filt


def get_interbasin_timeseries_modified(sst_da: xr.DataArray, detrend: bool,
                          base_start: str, base_end: str,
                          filtered=True):

    """Get the timeseries of the TPI index from SST data.

    Note that the climatological period is the entire time period spanned by
    the input sst data. Must specify option to remove the linear trend.
    Default filtering is a 13-year low pass Chebyshev filter.

    Args:
        sst_da (xr.DataArray): monthly SST data array
        linear_detrend (bool, default: True): if True, remove linear trend
        filtered (bool, default: True): if True, apply 13-year lowpass filter

    Returns:
        interbasin_idx (xr.DataArray): timeseries of the interbasin index in units of
        temperature

    """

    # Detrend the data (optional)
    if detrend:

        # Detrend by removing timeseries of global mean SSTs
        global_mean_ssts = utils.calc_cos_wmean(sst_da)
        sst_da = sst_da-global_mean_ssts

        # Detrend by removing linear trend (does not work for IOB)
        # sst_da = utils.detrend_array(sst_da, "time", 1)

    # Calculate SST anomalies by removing monthly climatology
    # (default climatological period is the entire data period)
    da_anm = utils.remove_monthly_clm(sst_da, base_start, base_end)

    # Get slice of SST anomalies for each interbasin region
    anm1, anm2, anm3 = _get_interbasin_areas_modified(da_anm)

    # Take area-weighted averaged of each region
    mean1 = utils.calc_cos_wmean(anm1)
    mean2 = utils.calc_cos_wmean(anm2)
    mean3 = utils.calc_cos_wmean(anm3)

    # Calculate interbasin index
    interbasin_idx = _apply_interbasin_formula(mean1, mean2, mean3)
    
    # Apply 13-year Chebyshev filter (optional)
    if filtered is True:
        interbasin_idx = _apply_cheby1_filter(interbasin_idx)

    return interbasin_idx


def find_positive_negative_periods(interbasin_idx, period_length=13, start_year='1900', end_year='2014'):
    """
    Finds periods where the interbasin index is consistently positive or negative.

    Parameters:
        interbasin_idx (xarray.DataArray): The interbasin index with 'ensemble' and 'time' dimensions.
        period_length (int): The length of the period to check for consistency (default is 13 years).
        start_year (str): Start year for the time slice.
        end_year (str): End year for the time slice.

    Returns:
        tuple of pd.DataFrame: DataFrames for positive and negative periods.
    """
    # Slice the index for the specified time range
    index = interbasin_idx.sel(time=slice(start_year, end_year))
    
    # Initialize lists to store results
    positive_ensembles = []
    positive_time_starts = []
    negative_ensembles = []
    negative_time_starts = []
    
    # Number of ensembles and time steps
    num_ensembles = index.sizes['ensemble']
    num_time_steps = index.sizes['time']
    
    # Loop over each ensemble member
    for ensemble in range(num_ensembles):
        data = index.isel(ensemble=ensemble).values  # Extract data for the ensemble member
        
        # Loop over possible starting points for the period
        for t in range(num_time_steps - period_length + 1):
            period = data[t:t + period_length]
            
            # Check if all values in the period are positive
            if np.all(period > 0):
                positive_ensembles.append(ensemble)
                positive_time_starts.append(index.coords['time'][t].dt.year.values)
            # Check if all values in the period are negative
            elif np.all(period < 0):
                negative_ensembles.append(ensemble)
                negative_time_starts.append(index.coords['time'][t].dt.year.values)
    
    # Create DataFrames from the results
    df_positive = pd.DataFrame({
        'Members': positive_ensembles,
        'Years': positive_time_starts
    })

    df_negative = pd.DataFrame({
        'Members': negative_ensembles,
        'Years': negative_time_starts
    })

    return df_positive, df_negative


def select_random_periods(interbasin_idx, period_length=13, start_year='1900', end_year='2014'):
    positive_index_df, negative_index_df = find_positive_negative_periods(interbasin_idx)

    num_rows_pos = positive_index_df.shape[0]
    num_rows_neg = negative_index_df.shape[0]

    random_ensembles_pos = np.random.randint(0, 100, size=num_rows_pos)
    random_ensembles_neg = np.random.randint(0, 100, size=num_rows_neg)

    random_year_starts_pos = np.random.randint(1900, 2003, size=num_rows_pos)
    random_year_starts_neg = np.random.randint(1900, 2003, size=num_rows_neg)

    random_index_len_pos_df = pd.DataFrame({
        'Members': random_ensembles_pos,
        'Years': random_year_starts_pos
    })

    random_index_len_neg_df = pd.DataFrame({
        'Members': random_ensembles_neg,
        'Years': random_year_starts_neg
    })

    return random_index_len_pos_df, random_index_len_neg_df
