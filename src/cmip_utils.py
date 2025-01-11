"""
cmip_utils.py
===================================================================

Functions for dealing with cmip6 output. 

"""

import xarray as xr

from src import utils
from src.inputs import *

ds_out = xr.open_mfdataset(tos_files['E3SM-1-0'])

def process_cmip_ensemble(model, anomaly=True):
    """Open CMIP6 data and regrid to common 1x1 gr grid.
    For each CMIP6 model, calculate MAM prect in msea region (precip_anm)
    and calculate corresponding DJF sst in Niño3.4 region (nino34_djf_ersst)"""

    ds_tos = utils.get_cmip6_da(tos_files[model], ds_out)
    ds_pr = utils.get_cmip6_da(pr_files[model], ds_out)

    da_tos = ds_tos['tos'].sel(time=slice('1900','2014'))
    da_pr = ds_pr['pr'].sel(time=slice('1900','2014'))
    da_pr *= 30 * 24 * 60 * 60  # kg m-2 s-1 to mm/month

    if anomaly==True:
        precip_anm = utils.get_cmip_msea_prect_anomaly_timeseries_mam(da_pr)
    else:
        precip_anm = utils.get_cmip_msea_prect_climatology_timeseries_mam(da_pr)
    nino34_djf_ersst = utils.get_cmip_nino34_sst_anomaly_timeseries_djf(da_tos)

    return nino34_djf_ersst, precip_anm
