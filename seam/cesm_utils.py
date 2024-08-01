# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: SEAM-env
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import pandas as pd
import xarray as xr
import cftime
from seam import utils

# %%
try:
    import xesmf as xe
except ImportError:
    xe = None

# %%
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

def convert_mms_to_mmmonth(da):
    seconds_in_month = [2678400, 2419200, 2678400, 2592000, 2678400, 2592000, 2678400,
        2678400, 2592000, 2678400, 2592000, 2678400]

    # Create a pandas DatetimeIndex object with month frequency for the years 2000-2011
    time_index = pd.date_range('2000-01', '2001-01', freq='M')

    # Create an xarray DataArray with the seconds_in_month data and time_index as coordinates
    seconds_in_month_da = xr.DataArray(seconds_in_month, dims='time', coords={'time': time_index})

    # Rename the coordinates to use month numbers instead of names
    seconds_in_month_da = seconds_in_month_da.rename({'time': 'month'})
    seconds_in_month_da['month'] = np.arange(1, 13)

    # Multiply m/s by s/month
    new_da0 = da.groupby('time.month')*seconds_in_month_da

    # Multiply m by mm/m
    new_da = new_da0*1000

    return new_da


def get_PRECT_da(project, ens_num, mask_and_convert=True):
    if project=="piControl":
        PRECT_dir = "/net/fs01/data/CESM2_projects/CESM2-CMIP6-piControl/atm/monthly_ave/PRECT/"
        PRECT_file = f"b.e21.B1850.f09_g17.CMIP6-piControl.001.cam.h0.PRECT.{ens_num}.nc"
        precip_ds = utils.get_ds(PRECT_dir + PRECT_file, cesm=True)
        PRECT_da = precip_ds["PRECT"]
    
    if project=="ATL_pacemaker":
        PRECC_dir = "/home/eleroy/proj-dirs/SEAM/data/ExtData/CESM1_ATL_pacemaker/atm/monthly_ave/PRECC/"
        PRECC_file = f"b.e11.B20TRLENS.f09_g16.SST.restoring.NATL.1920.ens{ens_num}.cam.h0.PRECC.192001-201312.nc"
        PRECC_ds0 = utils.get_ds(PRECC_dir + PRECC_file, cesm=True)
        PRECC_da = PRECC_ds0["PRECC"]
        PRECL_dir = "/home/eleroy/proj-dirs/SEAM/data/ExtData/CESM1_ATL_pacemaker/atm/monthly_ave/PRECL/"
        PRECL_file = f"b.e11.B20TRLENS.f09_g16.SST.restoring.NATL.1920.ens{ens_num}.cam.h0.PRECL.192001-201312.nc"
        PRECL_ds0 = utils.get_ds(PRECL_dir + PRECL_file, cesm=True)
        PRECL_da = PRECL_ds0["PRECL"]
        PRECT_da = PRECC_da + PRECL_da

    if project=="IOD_pacemaker":
        # PRECC = convective precipitation
        PRECC_dir = f"/home/eleroy/proj-dirs/SEAM/data/ExtData/CESM1_IOD_pacemaker/atm/monthly_ave/PRECC/"
        PRECC_file1 = f"b.e11.B20TRLENS.f09_g16.SST.rstor.IOD.ens{ens_num}.cam.h0.PRECC.192001-200512.nc"
        PRECC_file2 = f"b.e11.BRCP85LENS.f09_g16.SST.rstor.IOD.ens{ens_num}.cam.h0.PRECC.200601-201312.nc"
        PRECC_ds_1 = utils.get_ds(PRECC_dir + PRECC_file1, cesm=True)
        PRECC_ds_2 = utils.get_ds(PRECC_dir + PRECC_file2, cesm=True)
        PRECC_da_1 = PRECC_ds_1["PRECC"]
        PRECC_da_2 = PRECC_ds_2["PRECC"]
        PRECC_da = xr.concat([PRECC_da_1, PRECC_da_2], dim='time')
        PRECL_dir = f"/home/eleroy/proj-dirs/SEAM/data/ExtData/CESM1_IOD_pacemaker/atm/monthly_ave/PRECL/"
        PRECL_file1 = f"b.e11.B20TRLENS.f09_g16.SST.rstor.IOD.ens{ens_num}.cam.h0.PRECL.192001-200512.nc"
        PRECL_file2 = f"b.e11.BRCP85LENS.f09_g16.SST.rstor.IOD.ens{ens_num}.cam.h0.PRECL.200601-201312.nc"
        PRECL_ds_1 = utils.get_ds(PRECL_dir + PRECL_file1, cesm=True)
        PRECL_ds_2 = utils.get_ds(PRECL_dir + PRECL_file2, cesm=True)
        PRECL_da_1 = PRECL_ds_1["PRECL"]
        PRECL_da_2 = PRECL_ds_2["PRECL"]
        PRECL_da = xr.concat([PRECL_da_1, PRECL_da_2], dim='time')
        PRECT_da = PRECC_da + PRECL_da
    
    if project=="PAC_pacemaker":
        # PRECC = convective precipitation
        PRECC_dir = f"/home/eleroy/proj-dirs/SEAM/data/ExtData/CESM1_PAC_pacemaker/atm/monthly_ave/PRECC/"
        PRECC_file1 = f"b.e11.B20TRLENS.f09_g16.SST.restoring.ens{ens_num}.cam.h0.PRECC.192001-200512.nc"
        PRECC_file2 = f"b.e11.BRCP85LENS.f09_g16.SST.restoring.ens{ens_num}.cam.h0.PRECC.200601-201312.nc"
        PRECC_ds_1 = utils.get_ds(PRECC_dir + PRECC_file1, cesm=True)
        PRECC_ds_2 = utils.get_ds(PRECC_dir + PRECC_file2, cesm=True)
        PRECC_da_1 = PRECC_ds_1["PRECC"] 
        PRECC_da_2 = PRECC_ds_2["PRECC"]
        PRECC_da = xr.concat([PRECC_da_1, PRECC_da_2], dim='time')
        PRECL_dir = f"/home/eleroy/proj-dirs/SEAM/data/ExtData/CESM1_PAC_pacemaker/atm/monthly_ave/PRECL/"
        PRECL_file1 = f"b.e11.B20TRLENS.f09_g16.SST.restoring.ens{ens_num}.cam.h0.PRECL.192001-200512.nc"
        PRECL_file2 = f"b.e11.BRCP85LENS.f09_g16.SST.restoring.ens{ens_num}.cam.h0.PRECL.200601-201312.nc"
        PRECL_ds_1 = utils.get_ds(PRECL_dir + PRECL_file1, cesm=True)
        PRECL_ds_2 = utils.get_ds(PRECL_dir + PRECL_file2, cesm=True)
        PRECL_da_1 = PRECL_ds_1["PRECL"]
        PRECL_da_2 = PRECL_ds_2["PRECL"]
        PRECL_da = xr.concat([PRECL_da_1, PRECL_da_2], dim='time')
        PRECT_da = PRECC_da + PRECL_da

    if mask_and_convert:
        PRECT_da = utils.mask_ocean(PRECT_da)
        PRECT_da = convert_mms_to_mmmonth(PRECT_da)
    
    return PRECT_da

def get_SST_da(project, ens_num):
    if project=="piControl":
        SST_dir = "/net/fs01/data/CESM2_projects/CESM2-CMIP6-piControl/ocn/monthly_ave/SST/"
        SST_file = f"b.e21.B1850.f09_g17.CMIP6-piControl.001.pop.h.SST.{ens_num}.nc"
        ds = utils.get_ds(SST_dir + SST_file, cesm=True)
        SST_da = ds["SST"].isel(z_t=0)
        return SST_da

    if project=="LENS":
        SST_dir = "/net/fs01/data/CESM2_projects/CESM2-LE/ocn/monthly_ave/SST/"
        SST_file = f"b.e21.BHISTsmbb.f09_g17.LE2-1301.020.pop.h.SST.199001-199912.nc"
        ds = utils.get_ds(SST_dir + SST_file, cesm=True)
        SST_da = ds["SST"].isel(z_t=0)
        return SST_da
    
    if project=="ATL_pacemaker":
        SST_dir = "/home/eleroy/proj-dirs/SEAM/data/ExtData/CESM1_ATL_pacemaker/ocn/monthly_ave/SST/"
        SST_file = f"b.e11.B20TRLENS.f09_g16.SST.restoring.NATL.1920.ens{ens_num}.pop.h.SST.192001-201312.nc"
        ds = utils.get_ds(SST_dir + SST_file, cesm=True)
        SST_da = ds["SST"].isel(z_t=0)
        return SST_da

    if project=="IOD_pacemaker":
        SST_dir = f"/home/eleroy/proj-dirs/SEAM/data/ExtData/CESM1_IOD_pacemaker/ocn/monthly_ave/SST/"
        SST_file1 = f"b.e11.B20TRLENS.f09_g16.SST.rstor.IOD.ens{ens_num}.pop.h.SST.192001-200512.nc"
        SST_file2 = f"b.e11.BRCP85LENS.f09_g16.SST.rstor.IOD.ens{ens_num}.pop.h.SST.200601-201312.nc"
        SST_ds_1 = utils.get_ds(SST_dir + SST_file1, cesm=True)
        SST_ds_2 = utils.get_ds(SST_dir + SST_file2, cesm=True)
        SST_da_1 = SST_ds_1["SST"].isel(z_t=0)
        SST_da_2 = SST_ds_2["SST"].isel(z_t=0)
        SST_da = xr.concat([SST_da_1, SST_da_2], dim='time')
        return SST_da
    
    if project=="PAC_pacemaker":
        SST_dir = f"/home/eleroy/proj-dirs/SEAM/data/ExtData/CESM1_PAC_pacemaker/ocn/monthly_ave/SST/"
        SST_file1 = f"b.e11.B20TRLENS.f09_g16.SST.restoring.ens{ens_num}.pop.h.SST.192001-200512.nc"
        SST_file2 = f"b.e11.BRCP85LENS.f09_g16.SST.restoring.ens{ens_num}.pop.h.SST.200601-201312.nc"
        SST_ds_1 = utils.get_ds(SST_dir + SST_file1, cesm=True)
        SST_ds_2 = utils.get_ds(SST_dir + SST_file2, cesm=True)
        SST_da_1 = SST_ds_1["SST"].isel(z_t=0)
        SST_da_2 = SST_ds_2["SST"].isel(z_t=0)
        SST_da = xr.concat([SST_da_1, SST_da_2], dim='time')


        return SST_da

