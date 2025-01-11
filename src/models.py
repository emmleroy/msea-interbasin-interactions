"""
models.py
===================================================================

Contains unique identifiers for model ensemble members

"""

# CESM-LENS2 ensemble members
CESM2_ensemble_members = ['LE2-1001.001', 'LE2-1011.001', 'LE2-1021.002', 'LE2-1031.002', 'LE2-1041.003', 'LE2-1051.003',
            'LE2-1061.004', 'LE2-1071.004', 'LE2-1081.005', 'LE2-1091.005', 'LE2-1101.006', 'LE2-1111.006',
            'LE2-1121.007', 'LE2-1131.007', 'LE2-1141.008', 'LE2-1151.008', 'LE2-1161.009', 'LE2-1171.009',
            'LE2-1181.010', 'LE2-1191.010', 
            'LE2-1231.001', 'LE2-1231.002', 'LE2-1231.003', 'LE2-1231.004',
            'LE2-1231.005', 'LE2-1231.006', 'LE2-1231.007', 'LE2-1231.008', 'LE2-1231.009', 'LE2-1231.010',
            'LE2-1231.011', 'LE2-1231.012', 'LE2-1231.013', 'LE2-1231.014', 'LE2-1231.015', 'LE2-1231.016',
            'LE2-1231.017', 'LE2-1231.018', 'LE2-1231.019', 'LE2-1231.020',
            'LE2-1251.001', 'LE2-1251.002', 'LE2-1251.003', 'LE2-1251.004',
            'LE2-1251.005', 'LE2-1251.006', 'LE2-1251.007', 'LE2-1251.008', 'LE2-1251.009', 'LE2-1251.010',
            'LE2-1251.011', 'LE2-1251.012', 'LE2-1251.013', 'LE2-1251.014', 'LE2-1251.015', 'LE2-1251.016',
            'LE2-1251.017', 'LE2-1251.018', 'LE2-1251.019', 'LE2-1251.020',
            'LE2-1301.001', 'LE2-1301.002', 'LE2-1301.003', 'LE2-1301.004', 'LE2-1301.005', 'LE2-1301.006',
            'LE2-1301.007', 'LE2-1301.008', 'LE2-1301.009', 'LE2-1301.010', 'LE2-1301.011', 'LE2-1301.012', 
            'LE2-1301.013', 'LE2-1301.014', 'LE2-1301.015', 'LE2-1301.016', 'LE2-1301.017', 'LE2-1301.018',
            'LE2-1301.019', 'LE2-1301.020',
            'LE2-1281.001', 'LE2-1281.002', 'LE2-1281.003', 'LE2-1281.004', 'LE2-1281.005', 'LE2-1281.006',
            'LE2-1281.007', 'LE2-1281.008', 'LE2-1281.009', 'LE2-1281.010', 'LE2-1281.011', 'LE2-1281.012',
            'LE2-1281.013', 'LE2-1281.014', 'LE2-1281.015', 'LE2-1281.016', 'LE2-1281.017', 'LE2-1281.018',
            'LE2-1281.019', 'LE2-1281.020']

# Corresponding List of CESM2-SSP2-7.0 Ensemble Members
# Documentation: https://www2.cesm.ucar.edu/working_groups/CVC/simulations/cesm2-ssp245.html
# Documentation: https://www2.cesm.ucar.edu/working_groups/CVC/simulations/cesm2-ssp585.html

CESM2_SSP245_ensemble_members = ['001', '002', '003', '004', '005',
           '006', '007', '008', '009', '010',
           '011', '012', '013', '014', '015',
            '016']

CESM2_SSP585_ensemble_members = ['001', '002', '003', '004', '005',
           '006', '007', '008', '009', '010',
           '011', '012', '013', '014', '015']

# LENS2 members corresponding to SSP245 16-member ensemble
LENS2_members_for_CESM2_SSP245 = ['LE2-1231.011',
           'LE2-1231.012',
           'LE2-1231.013',
           'LE2-1231.014',
           'LE2-1231.015',
           'LE2-1231.016',
           'LE2-1231.017',
           'LE2-1231.018',
           'LE2-1251.012',
           'LE2-1251.013',
           'LE2-1251.014',
           'LE2-1251.015',
           'LE2-1251.016',
           'LE2-1251.017',
           'LE2-1251.018',
           'LE2-1251.011']

# LENS2 members corresponding to SSP585 15-member ensemble
LENS2_members_for_CESM2_SSP585 = ['LE2-1011.001',
           'LE2-1031.002',
           'LE2-1051.003',
           'LE2-1071.004',
           'LE2-1091.005',
           'LE2-1111.006',
           'LE2-1131.007',
           'LE2-1151.008',
           'LE2-1171.009',
           'LE2-1191.010',
           'LE2-1251.011',
           'LE2-1251.012',
           'LE2-1251.013',
           'LE2-1251.014',
           'LE2-1251.015']

CMIP6_models = ['ACCESS-CM2', 
          'ACCESS-ESM1-5', 
          'BCC-CSM2-MR', 
          'BCC-ESM1',
          'CAMS-CSM1-0',
          'CanESM5', 
          'CAS-ESM2-0',
          'CESM2-FV2',
          'CESM2-WACCM-FV2', 
          #'CESM2-WACCM',
          'CIESM', 
          'E3SM-1-0', 
          'E3SM-1-1-ECA',
          'E3SM-1-1', 
          'E3SM-2-0',
          'E3SM-2-0-NARRM',
          'EC-Earth3-AerChem',
          'EC-Earth3-CC', 
          'EC-Earth3', 
          'EC-Earth3-Veg-LR', 
          'FGOALS-g3', 
          'FGOALS-f3-L',
          'FIO-ESM-2-0', 
          'GFDL-CM4',
          'GFDL-ESM4', 
          'GISS-E2-1-G', 
          'GISS-E2-1-H',
          'GISS-E2-2-H',
          #'IITM-ESM',
          'INM-CM4-8', 
          'INM-CM5-0', 
          'IPSL-CM6A-LR', 
          'KACE-1-0-G',
           #'KIOST-ESM',
          'MCM-UA-1-0', 
          'MIROC6', 
          'MPI-ESM-1-2-HAM', 
          'MPI-ESM1-2-HR', 
          'MPI-ESM1-2-LR',
          'MRI-ESM2-0', 
          'NESM3', 
          'NorCPM1',
          'NorESM2-LM', 
          'NorESM2-MM', 
          'SAM0-UNICON']