"""
cesm_utils.py
===================================================================

Functions for dealing with CESM large ensemble output. 

"""

from concurrent.futures import ProcessPoolExecutor

import cftime
import numpy as np
import xarray as xr

from src import utils
from src.inputs import *

try:
    import xesmf as xe
except ImportError:
    xe = None

def regrid_cesm(
    ds,
    d_lon_lat_kws={"lon": 5, "lat": 5},
    method="bilinear",
    periodic=False,
    filename=None,
    reuse_weights=True,
    tsmooth_kws=None,
    how=None,
):
    """
    Copied from https://climpred.readthedocs.io/en/stable/_modules/climpred/smoothing.html
    Quick regridding function. Adapted from
    https://github.com/JiaweiZhuang/xESMF/pull/27/files#diff-b537ef68c98c2ec11e64e4803fe4a113R105.
    Args:
        ds (xarray-object): Contain input and output grid coordinates.
            Look for variables ``lon``, ``lat``, and optionally ``lon_b``,
            ``lat_b`` for conservative method.
            Shape can be 1D (Nlon,) and (Nlat,) for rectilinear grids,
            or 2D (Ny, Nx) for general curvilinear grids.
            Shape of bounds should be (N+1,) or (Ny+1, Nx+1).
        d_lon_lat_kws (dict): optional
            Longitude/Latitude step size (grid resolution); if not provided,
            lon will equal 5 and lat will equal lon
            (optional)
        method (str): Regridding method. Options are:
            - 'bilinear'
            - 'conservative', **need grid corner information**
            - 'patch'
            - 'nearest_s2d'
            - 'nearest_d2s'
        periodic (bool): Periodic in longitude? Default to False. optional
            Only useful for global grids with non-conservative regridding.
            Will be forced to False for conservative regridding.
        filename (str): Name for the weight file. (optional)
            The default naming scheme is:
                {method}_{Ny_in}x{Nx_in}_{Ny_out}x{Nx_out}.nc
                e.g. bilinear_400x600_300x400.nc
        reuse_weights (bool) Whether to read existing weight file to save
            computing time. False by default. (optional)
        tsmooth_kws (None): leads nowhere but consistent with `temporal_smoothing`.
        how (None): leads nowhere but consistent with `temporal_smoothing`.
        Returns:
            ds (xarray.object) regridded
    """

    if xe is None:
        raise ImportError(
            "xesmf is not installed; see"
            "https://xesmf.readthedocs.io/en/latest/installation.html"
        )

    def _regrid_it(da, d_lon, d_lat, **kwargs):
        """
        Global 2D rectilinear grid centers and bounds
        Args:
            da (xarray.DataArray): Contain input and output grid coords.
                Look for variables ``lon``, ``lat``, ``lon_b``, ``lat_b`` for
                conservative method, and ``TLAT``, ``TLON`` for CESM POP grid
                Shape can be 1D (Nlon,) and (Nlat,) for rectilinear grids,
                or 2D (Ny, Nx) for general curvilinear grids.
                Shape of bounds should be (N+1,) or (Ny+1, Nx+1).
            d_lon (float): Longitude step size, i.e. grid resolution
            d_lat (float): Latitude step size, i.e. grid resolution
        Returns:
            da : xarray DataArray with coordinate values
        """

        def check_lon_lat_present(da):
            if method == "conservative":
                if "lat_b" in ds.coords and "lon_b" in ds.coords:
                    return da
                else:
                    raise ValueError(
                        'if method == "conservative", lat_b and lon_b are required.'
                    )
            else:
                if "lat" in ds.coords and "lon" in ds.coords:
                    return da
                elif "lat_b" in ds.coords and "lon_b" in ds.coords:
                    return da
                # for CESM POP grid
                elif "TLAT" in ds.coords and "TLONG" in ds.coords:
                    da = da.rename({"TLAT": "lat", "TLONG": "lon"})
                    return da
                else:
                    raise ValueError(
                        "lon/lat or lon_b/lat_b or TLAT/TLON not found, please rename."
                    )

        da = check_lon_lat_present(da)
        grid_out = {
            "lon": np.arange(da.lon.min(), da.lon.max() + d_lon, d_lon),
            "lat": np.arange(da.lat.min(), da.lat.max() + d_lat, d_lat),
        }
        regridder = xe.Regridder(da, grid_out, **kwargs)
        return regridder(da)

    # check if lon or/and lat missing
    if ("lon" in d_lon_lat_kws) and ("lat" in d_lon_lat_kws):
        pass
    elif ("lon" not in d_lon_lat_kws) and ("lat" in d_lon_lat_kws):
        d_lon_lat_kws["lon"] = d_lon_lat_kws["lat"]
    elif ("lat" not in d_lon_lat_kws) and ("lon" in d_lon_lat_kws):
        d_lon_lat_kws["lat"] = d_lon_lat_kws["lon"]
    else:
        raise ValueError("please provide either `lon` or/and `lat` in d_lon_lat_kws.")

    kwargs = {
        "d_lon": d_lon_lat_kws["lon"],
        "d_lat": d_lon_lat_kws["lat"],
        "method": method,
        "periodic": periodic,
        "filename": filename,
        "reuse_weights": reuse_weights,
    }

    ds = _regrid_it(ds, **kwargs)

    return ds


def time_set_midmonth(ds, time_name, deep=False):
    """
    Return copy of ds with values of ds[time_name] replaced with mid-month
    values (day=15) rather than end-month values.
    Copied from https://github.com/NCAR/iPOGS/blob/c2cca9f398546729dbf7b08fe0baf78aa9a3dcaf/notebooks/AMOCz_0.1deg_RCP_loop_time.ipynb

    """
    year = ds[time_name].dt.year
    month = ds[time_name].dt.month
    year = xr.where(month==1,year-1,year)
    month = xr.where(month==1,12,month-1)
    nmonths = len(month)
    newtime = [cftime.DatetimeNoLeap(year[i], month[i], 15) for i in range(nmonths)]
    ds[time_name] = newtime
    return ds


def process_cesm_member(ensemble_member, cesm_directory, file_suffix, anomaly=True):
    """ Open pre-processed LENS2 data of
    a single ensemble member. Files should be in the CESM_PROCESSED_DATA directory 
    and have filename: {ensemble_member}.{file_suffix}.nc"""

    ds = xr.open_dataset(f"{cesm_directory}/{ensemble_member}.{file_suffix}.nc")

    if file_suffix == "PRECT.MSEA":
        if anomaly==True: 
            ds = utils.get_cesm_msea_prect_anomaly_timeseries_mam(ds, months=[3,4,5], detrend_option=False)
        else:
            ds = utils.get_cesm_msea_prect_climatology_timeseries_mam(ds, months=[3,4,5], detrend_option=False)
    if file_suffix == "SST.Nino34":
        ds = utils.get_cesm_nino34_sst_anomaly_timeseries_djf(ds, detrend_option=False)

    return ds


def process_cesm_ensemble(ensemble_members, cesm_directory, file_suffix, anomaly=True):
    """ Open pre-processed LENS2 data from all ensemble members using 
    parallel processing. Files should be in the CESM_PROCESSED_DATA directory 
    and have filename: {ensemble_member}.{file_suffix}.nc"""

    processed_cesm_list = []
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(
            process_cesm_member, 
            ensemble_members,
            [cesm_directory]*len(ensemble_members),
            [file_suffix]*len(ensemble_members),
            [anomaly]*len(ensemble_members),
            ))
    for result in results:
        processed_cesm_list.append(result)
 
    processed_cesm_data = xr.concat(processed_cesm_list, dim='ensemble')

    if "z_t" in processed_cesm_data.dims:
        processed_cesm_data = processed_cesm_data.isel(z_t=0, drop=True)

    return processed_cesm_data


def calculate_cesm_member_runcorr(ensemble_member, cesm_directory, window):
    """Calculate running correlation between Niño3.4 and MSEA prect for 
    a single ensemble member, for a given window length."""
    
    #print(f"Processing {ens}")
    monthly_sst_da = xr.open_dataset(f"{cesm_directory}/{ensemble_member}.SST.Nino34.nc")
    sst_anm = utils.get_cesm_nino34_sst_anomaly_timeseries_djf(monthly_sst_da, detrend_option=False)
    
    prec_da = xr.open_dataset(f"{cesm_directory}/{ensemble_member}.PRECT.MSEA.nc")
    precip_anm = utils.get_cesm_msea_prect_anomaly_timeseries_mam(prec_da, months=[3,4,5], detrend_option=False)

    #print("Getting running correlation...")
    corr_lead = utils.get_running_corr(precip_anm, sst_anm.shift(time=1), window=window)
    
    return (corr_lead, precip_anm, sst_anm)
    

def calculate_cesm_ensemble_runcorr(ensemble_members, cesm_directory, window):
    """Calculate running correlation between Niño3.4 SSTs and MSEA prect for 
    all ensemble members in members, for a given window length."""
    lead_correlations = []
    all_precips = []
    all_ssts = []
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(
            calculate_cesm_member_runcorr,
            ensemble_members,
            [cesm_directory]*len(ensemble_members),
            [window]*len(ensemble_members),
            ))
    for result in results:
        corr_lead, precip_anm, sst_anm = result
        lead_correlations.append(corr_lead)
        all_precips.append(precip_anm)
        all_ssts.append(sst_anm)
    return lead_correlations, all_precips, all_ssts