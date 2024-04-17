#%%
import pandas as pd
from scipy.interpolate import griddata
from get_flood_raster_KSC import calculate_flooding_days, threshold_to_days, load_dem
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

#%%

## Make a tidal surface with vdatum

NOAA_SLR_DEM_PATH = './inputData/NOAA_SLR_DEM/NOAA_SLR_DEM_J995345.tif'
dem_xr = load_dem(NOAA_SLR_DEM_PATH) #
#%%
# Export a zeros raster in MHHW
zeros = xr.zeros_like(dem_xr)

# should have the same crs as the dem
zeros.rio.write_crs(dem_xr.rio.crs)

# should have the same attributes as the dem
zeros.attrs = dem_xr.attrs

#downsample the zeros raster for every 100m with boundary option
zeros = zeros.coarsen(x=100, y=100, boundary='pad').mean()

# export x,y arrays to points in csv
zeros = zeros.stack(z=('x', 'y'))

# export to csv, do not include 'spatial_ref' column
zeros.to_dataframe(name='zeros').drop(columns='spatial_ref').to_csv('./viz/zeros.csv', index=False)

#%%
## RUN VDATUM MANUALLY, THIS IS TO BE UPDATED
input_file = './viz/zeros.csv'
output_file = './tidal_surface/zeros_New.txt'
tidal_dir = './tidal_surface'
# #%% NEED TO FIGURE THIS PART OUT, DO noT Have all parameters right yet in the command line code
# import subprocess

# vdatum_cmd = 'vdatum -i:'+ input_file + ' -o:'+ output_file + '-ihorz:utm -ivert:MHHW -ohorz:utm -overt:NAVD88 -ell:grs80 -zone:17 -geoid:geoid18 -epoch:0.0'
# subprocess.run(vdatum_cmd, check=True, shell=True)

# %%
# %%
# make a tidal surface from the x,y,z file at ./tidal_surface/zeros_New.txt
csv_file = output_file
tidal_surface = pd.read_csv(csv_file, sep=',', header=None, names=['x', 'y', 'z'])

# make all z values less than -99999 to nan
tidal_surface.loc[tidal_surface['z'] == -999999, 'z'] = np.nan

# grid the points into the x,y grid from the dem
# make a meshgrid of the x,y values
x = np.linspace(dem_xr.x.min(), dem_xr.x.max(), dem_xr.x.size)
y = np.linspace(dem_xr.y.min(), dem_xr.y.max(), dem_xr.y.size)

X, Y = np.meshgrid(x, y)

# interpolate the z values onto the grid using nearest neighbor interpolation
Z = griddata(tidal_surface[['x', 'y']].values, tidal_surface['z'].values, (X, Y), method='linear')

# make a new xarray DataArray with the interpolated values
tidal_surface = xr.DataArray(
    Z,
    coords={'y': y, 'x': x},
    dims=['y', 'x'],
    attrs={'crs': dem_xr.rio.crs}
)

# smooth the tidal surface
from scipy.ndimage import gaussian_filter
tidal_surface = xr.DataArray(
    gaussian_filter(tidal_surface, sigma=50), #<--- equivalent of 50*3m = 150m smoothing
    coords={'y': y, 'x': x},
    dims=['y', 'x'],
    attrs={'crs': dem_xr.rio.crs}
)
# %%
# save the tidal surface as xarray

tidal_surface = tidal_surface.rio.write_crs(dem_xr.rio.crs)

#copy the crs from the dem_xr.rio.crs



tidal_surface.attrs = dem_xr.attrs

#%%
tidal_surface.to_netcdf('./tidal_surface/tidal_surface.nc')

#TODO: Do we need to save the tidal surface as a geotiff? Or netcdf? 
# Is the "graininess" coming from the interpolation method, 
#or from doing operations on the tif files?

# write the tidal surface to a geotiff
tidal_surface.rio.write_nodata(np.nan)
tidal_surface.rio.to_raster('./tidal_surface/tidal_surface.tif')

# %%