"""
Contains directories and filenames for data inputs
"""

import os
from src import models

cmip6_models = models.CMIP6_models

# Define path to ExtData and CESM processed data here:
EXT_DATA = "/home/eleroy/proj-dirs/SEAM/data/ExtData/"
CESM_PROCESSED_DATA = "/home/eleroy/proj-dirs/SEAM/data/analysis_data/CESM-LE2_cropped_data/"

sst_reanalysis_source_to_file = {
    "ERSST": f"{EXT_DATA}/ERSST/sst.mnmean.v5.nc",
    "HADISST": f"{EXT_DATA}/HadISST/HadISST_sst.nc",
    "COBESST": f"{EXT_DATA}/COBE_SST2/sst.mon.mean.nc"
}

prect_gridded_rain_gauge_source_to_file = {
    "GPCC": f"{EXT_DATA}/GPCC/full_v2020/precip.mon.total.0.5x0.5.v2020.nc",
    "CRUT": f"{EXT_DATA}/CRU_TS4.06/cru_ts4.06.1901.2021.pre.dat.nc",
    "APHR": f"{EXT_DATA}/APHRODITE/APHRO_MA_050deg_V1101_EXR1.1951-2015.mm_per_month.nc"
}

# Define CESM tos_files (SST) and pr_files (PRECT)
tos_dir = f'{EXT_DATA}/CMIP6/historical/r1i1p1f1/Omon/tos'
pr_dir = f'{EXT_DATA}/CMIP6/historical/r1i1p1f1/Amon/pr'

tos_files = {model: [] for model in cmip6_models}
pr_files = {model: [] for model in cmip6_models}

for file_name in os.listdir(tos_dir):
    for model in cmip6_models:
        if f"_{model}_" in file_name or file_name.endswith(f"_{model}.nc"):
            full_path = os.path.join(tos_dir, file_name)
            tos_files[model].append(full_path)

for file_name in os.listdir(pr_dir):
    for model in cmip6_models:
        if f"_{model}_" in file_name or file_name.endswith(f"_{model}.nc"):
            full_path = os.path.join(pr_dir, file_name)
            pr_files[model].append(full_path)
