#%%
"""
This script renames the json files in the input data directory to be in MHHW datum instead of NAVD.
It also takes into account that the NAVD provided files are relative to 100 cm BELOW NAVD88 datum.
It creates a new directory for the MHHW data and copies the files from the original directory, adjusting the thresholds accordingly.
"""

# Import necessary libraries
import math
import numpy as np
import json
import os
import xarray as xr
import math
# Constants
SCENARIOS = ['high', 'int', 'int_high', 'int_low', 'low']
YEARS = np.arange(2020, 2101, 10)  # Decadal intervals
FLOOD_DAYS_RASTER_DIR = './flood_days_raster'
TGDATA_DIR_COAST = './inputData/ensemble_stats/8721604'
TGDATA_DIR_INLAND = './inputData/ensemble_stats/02248380'

TGDATA_DIR_COAST_MHHW = './inputData/8721604_MHHW'
TGDATA_DIR_INLAND_MHHW = './inputData/02248380_MHHW'

TRIDENT_MHHW = 0.34  # meters MHHW from NOAA datum info (https://tidesandcurrents.noaa.gov/datums.html?datum=NAVD88&units=1&epoch=0&id=8721604&name=Trident+Pier%2C+Port+Canaveral&state=FL)

# Extract MHHW datum from MHHW surface
# Open netcdf file
mhhw_surface = xr.open_dataset('./tidal_surface/tidal_surface.nc')['__xarray_dataarray_variable__']

HauloverEastings = 523923
HauloverNorthings = 3175600 # this is about 3km south of the actual Haulover Inlet

HAULOVER_MHHW = mhhw_surface.sel(x=HauloverEastings, y=HauloverNorthings, method='nearest').values.round(2)

# Rest of the code...

# For every scenario, rename json files to be in MHHW datum instead of NAVD
# 
# 
# First make a new directory for the MHHW data
#  - ./inputData/8721604_MHHW
#  - ./inputData/02248380_MHHW

#check if the directories exist, if not, create them
if not os.path.exists(TGDATA_DIR_COAST_MHHW):
    os.makedirs(TGDATA_DIR_COAST_MHHW)
    # make directories for each scenario
    for scenario in SCENARIOS:
        os.makedirs(os.path.join(TGDATA_DIR_COAST_MHHW, scenario))
if not os.path.exists(TGDATA_DIR_INLAND_MHHW):
    os.makedirs(TGDATA_DIR_INLAND_MHHW)
    # make directories for each scenario
    for scenario in SCENARIOS:
        os.makedirs(os.path.join(TGDATA_DIR_INLAND_MHHW, scenario))

#%%

# For each scenario, rename the json files to be in MHHW datum instead of NAVD
for scenario in SCENARIOS:
    # For coastal gauge
    for threshold in np.arange(0, 4.65, 0.010):  # Thresholds from 0 to 3 meters
        # Load the json file
        with open(os.path.join(TGDATA_DIR_COAST, f'{scenario}/{int(threshold * 100):03d}.json')) as f:
            data = json.load(f)
        # Write the data to a new json file in the MHHW directory
        # rounded_threshold = math.floor((threshold - 1.0 - TRIDENT_MHHW)*100)/100  # Adjust threshold to MHHW from NAVD
        rounded_threshold = (threshold - 1.0 - TRIDENT_MHHW).round(2)  # Adjust threshold to MHHW from NAVD
        # Format the file name to avoid rounding issues
        formatted_threshold = f"{rounded_threshold:.2f}".replace('.', '').zfill(3)
        with open(os.path.join(TGDATA_DIR_COAST_MHHW, f'{scenario}/{formatted_threshold}.json'), 'w') as f:
            json.dump(data, f)
#%%

for scenario in SCENARIOS:
    # For inland gauge
    for threshold in np.arange(0, 4.65, 0.010):  # Thresholds from 0 to 3 meters
        # Load the json file
        with open(os.path.join(TGDATA_DIR_INLAND, f'{scenario}/{int(threshold * 100):03d}.json')) as f:
            data = json.load(f)
        # Write the data to a new json file in the MHHW directory
        rounded_threshold = (threshold - 1.0 - HAULOVER_MHHW).round(2)  # Adjust threshold to MHHW from NAVD

        # Format the file name to avoid rounding issues
        formatted_threshold = f"{rounded_threshold:.2f}".replace('.', '').zfill(3)
        
        with open(os.path.join(TGDATA_DIR_INLAND_MHHW, f'{scenario}/{formatted_threshold}.json'), 'w') as f:
            json.dump(data, f)
# %%
