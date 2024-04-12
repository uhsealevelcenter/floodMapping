#%%
from get_flood_raster_KSC import calculate_flooding_days, threshold_to_days, load_dem
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

#**********************TESTING AREA********************************
#%%
#TEST 1: plot the flooding days by threshold
#plot totals by threshold, with x-axis as threshold and y-axis as total days flooded

scenario = 'int_low'
df = threshold_to_days(scenario)
plt.plot(df.loc[2100])
plt.xlabel('Threshold (m)')
plt.ylabel('Total Days Flooded')
# only look at 0-1m for now
plt.xlim(0, 1)

# make a vertical line for the 2ft mark
plt.axvline(0.61, color='r', linestyle='--')

#%%
## TEST 2: make a dummy dem_xr to test
dem_xr = xr.DataArray(
    np.array([[0.48, 0.2, 0.61, 0.83], [0.68, 0.45, 0.61, 0.37], [0.48, 0.2, 0, 0.4], [0.48, 0.2, 0.61, 0.6]]),
    coords={'y': np.arange(4), 'x': np.arange(4)},
    dims=['y', 'x'],
    attrs={'crs': 'EPSG:26917'}
)

mhhw_xr = xr.DataArray(
    np.full((4,4),0.3),
    coords={'y': np.arange(4), 'x': np.arange(4)},
    dims=['y', 'x'],
    attrs={'crs': 'EPSG:26917'}
)

scenario = 'int_low'
year = 2100
flooding_days = calculate_flooding_days(dem_xr, mhhw_xr, scenario, year)

#plot the flooding days
plt.imshow(flooding_days)
plt.colorbar()
#%% TEST 3: 
NOAA_SLR_DEM_PATH = './inputData/NOAA_SLR_DEM/NOAA_SLR_DEM_J995345.tif'
TIDAL_SURFACE_PATH = './tidal_surface/tidal_surface.tif'
dem_xr = load_dem(NOAA_SLR_DEM_PATH)
mhhw_xr = load_dem(TIDAL_SURFACE_PATH)
mhhw_xr_aligned = mhhw_xr.rio.reproject_match(dem_xr)

scenario = 'int_low'
year = 2100
scenario_dir = f'./flood_days_raster/{scenario}'

flooding_days = calculate_flooding_days(dem_xr, mhhw_xr_aligned, scenario, year)
flooding_days.rio.to_raster(os.path.join(scenario_dir, f'{year}.tif'))

#open this raster in QGIS to check stuff, OR:

#plot the flooding days
plt.imshow(flooding_days)
plt.colorbar()
# %%
