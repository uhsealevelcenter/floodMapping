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

xcoords = np.arange(dem_xr.coords['x'].min(), dem_xr.coords['x'].max(), 200)
ycoords = np.arange(dem_xr.coords['y'].min(), dem_xr.coords['y'].max(), 200)

zeros = np.zeros((len(ycoords), len(xcoords)))
zeros = xr.DataArray(
    zeros,
    coords={'y': ycoords, 'x': xcoords},
    dims=['y', 'x'],
    attrs={'crs': dem_xr.rio.crs}
)

# %%

# #downsample the zeros raster for every 100*3m with boundary option
# zeros = zeros.coarsen(x=100, y=100, boundary='pad').mean()

#%%

# check min and max of non-padded array
min_x = zeros['x'].min().item()
max_x = zeros['x'].max().item()

print(f"Min x coordinate in non-padded zeros array: {min_x}")
print(f"Max x coordinate in non-padded zeros array: {max_x}")

#check the min and max y coords of the dem_xr array
min_x = dem_xr['x'].min().item()
max_x = dem_xr['x'].max().item()

print(f"Min x coordinate in dem_xr array: {min_x}")
print(f"Max x coordinate in dem_xr array: {max_x}")
#%%
# export x,y arrays to points in csv
zeros = zeros.stack(z=('x', 'y'))

# export to csv, do not include 'spatial_ref' column
zeros.to_dataframe(name='zeros').to_csv('./viz/zeros.csv', index=False)

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
tidal_surface = pd.read_csv(csv_file, sep=',', header=1, names=['x', 'y', 'z'])
# make all z values less than -99999 to nan
tidal_surface.loc[tidal_surface['z'] == -999999, 'z'] = np.nan

# %% TEST FIG: SCATTER PLOT OF TIDAL SURFACE
plt.scatter(tidal_surface['x'], tidal_surface['y'], c=tidal_surface['z'], cmap='viridis')
plt.colorbar(label='z values')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter plot of tidal surface data')
# set aspect ratio to be equal
plt.gca().set_aspect('equal', adjustable='box')

#%%
# Create a target grid for interpolation
x_target = np.arange(dem_xr.coords['x'].min(), dem_xr.coords['x'].max(), 100)
y_target = np.arange(dem_xr.coords['y'].min(), dem_xr.coords['y'].max(), 100)
xv, yv = np.meshgrid(x_target, y_target)

#%%
# Interpolate using griddata
tidal_surface_interp = griddata(
    (tidal_surface['x'], tidal_surface['y']),  # Known data points
    tidal_surface['z'],  # Known data values
    (xv, yv),  # Target grid
    method='linear'
)
#%%
# Create a new xarray DataArray with the interpolated values
tidal_surface_xr = xr.DataArray(
    tidal_surface_interp,
    coords={'y': y_target, 'x': x_target},
    dims=['y', 'x'],
    attrs={'crs': dem_xr.rio.crs}
)

tidal_surface_xr = tidal_surface_xr.rio.write_crs(dem_xr.rio.crs)
#%%
# Fill known values into the DataArray
for _, row in tidal_surface.iterrows():
    tidal_surface_xr.loc[dict(y=row['y'], x=row['x'])] = row['z']

# Create a mask of the original NaN values
# nan_mask = tidal_surface_xr<0

# load in the mask for the shoreline 
inland_mask = xr.open_dataarray('./mask_3000_inland.nc')
# Set the CRS for inland_mask
inland_mask = inland_mask.astype(int).rio.write_crs(dem_xr.rio.crs)
# reproject inland mask to the tidal_surface_interpolated
inland_mask = inland_mask.rio.reproject_match(tidal_surface_xr)




# Combine original data with interpolated values
# tidal_surface_combined = tidal_surface_xr.where(nan_mask, tidal_surface_interp)

#%% plot
#make a smaller dem_xr to match the x,y of the tidal_surface_combined
dem_xr_subset = dem_xr.interp(x=tidal_surface_xr.x, y=tidal_surface_xr.y, method='nearest')
fig, ax = plt.subplots(figsize=(10, 10))
tidal_surface_xr.where((dem_xr_subset<0) & (tidal_surface_xr<0)).plot(ax=ax)

# fit tidal_surface_xr.where((dem_xr_subset<0) & (tidal_surface_xr<0)) to a plane and interpolate the rest of the values
# %%
# Fit a plane to the tidal surface
from scipy.optimize import curve_fit

# Function to define the plane
def plane(coords, a, b, c):
    x, y = coords
    return a * x + b * y + c

# Apply the condition once to the dataset
filtered_data = tidal_surface_xr.where((dem_xr_subset<0) & (tidal_surface_xr<0))

# Now extract x, y, and z values from the filtered dataset
x = filtered_data.x.values
y = filtered_data.y.values
z = filtered_data.values

# Create meshgrid from x and y
X, Y = np.meshgrid(x, y)

# Flatten the meshgrid and z values
x_flat = X.flatten()
y_flat = Y.flatten()
z_flat = z.flatten()

# Filter out NaN values to ensure x, y, and z have the same length
mask = ~np.isnan(z_flat)
x_valid = x_flat[mask]
y_valid = y_flat[mask]
z_valid = z_flat[mask]

# Ensure x and y are passed as separate arguments to curve_fit
# Combine x_valid and y_valid as a 2D array with shape (2, number_of_points)
coords = np.vstack((x_valid, y_valid))

# Fit the plane to the cleaned data
popt, _ = curve_fit(plane, coords, z_valid)

# Create a new z array with the plane values
# z_plane = plane((x_valid, y_valid), *popt)
z_plane = plane((X, Y), *popt)

# Create a new xarray DataArray with the plane values
# Since x_valid and y_valid are 1D, reshape them to match the original shape
# z_plane_reshaped = np.full(z.shape, np.nan)
# z_plane_reshaped[mask.reshape(z.shape)] = z_plane
#%%
tidal_surface_plane = xr.DataArray(
    z_plane,
    coords=filtered_data.coords,
    dims=filtered_data.dims,
    attrs={'crs': dem_xr.rio.crs}  # Assuming dem_xr is defined elsewhere
)

# Plot the plane
fig, ax = plt.subplots(figsize=(10, 10))
tidal_surface_plane.plot(ax=ax)
plt.show()
#%%
# fill in the positive values of tidal_surface_xr to the tidal_surface_plane
tidal_surface_filled = tidal_surface_plane.where((tidal_surface_xr<=0) | (np.isnan(tidal_surface_xr)), tidal_surface_xr)

# keep the nan area on the cape
tidal_surface_filled = tidal_surface_filled.where(~(np.isnan(tidal_surface_xr) & (tidal_surface_filled.x > 538000)), np.nan)

#interpolate tidal_surface_filled to the same coordinates as the dem_xr
tidal_surface_filled = tidal_surface_filled.interp(x=dem_xr.x, y=dem_xr.y, method='nearest')

# Plot the filled tidal surface
fig, ax = plt.subplots(figsize=(10, 10))
tidal_surface_filled.plot(ax=ax)

#%%
# interpolate missing values
tidal_surface_filled = tidal_surface_filled.interpolate_na(dim='x', method='linear')
#%% Plot the filled tidal surface
fig, ax = plt.subplots(figsize=(10, 10))
tidal_surface_filled.plot(ax=ax)


#%% smooth the tidal surface
from scipy.ndimage import gaussian_filter
tidal_surfaceFilt = xr.DataArray(
    gaussian_filter(tidal_surface_filled, sigma=50, mode='reflect'), #<--- equivalent of 50*3m = 150m smoothing
    coords={'y': dem_xr.y, 'x': dem_xr.x},
    dims=['y', 'x'],
    attrs={'crs': dem_xr.rio.crs}
)


#%%

fig, ax = plt.subplots(figsize=(10, 10))
tidal_surfaceFilt.plot(ax=ax)
# now make the tidal_surface_interpolated the same size as the inland_mask

#%%



# #%%make figure
# fig, ax = plt.subplots(figsize=(10, 10))


# tidal_surface_xr.plot(ax=ax)
# inland_mask.plot(ax=ax, alpha=0.5)

# # add shoreline
# # plt.plot(shoreline, shoreline.y, 'r--', label='Shoreline')

# #%%
# # Fill NaNs using nearest neighbor interpolation
# tidal_surface_xr = griddata(
#     (tidal_surface['x'], tidal_surface['y']),
#     tidal_surface['z'],
#     (xv, yv),
#     method='nearest'
# )

# #%%
# # Combine linear interpolation and nearest neighbor filling
# final_tidal_surface = np.where(np.isnan(tidal_surface_interp), tidal_surface_filled, tidal_surface_interp)

# #%%
# # look at tidal_surface_interp
# plt.imshow(tidal_surface_filled, cmap='viridis')

# #%%

# # Create a new xarray DataArray with the interpolated values
# tidal_surface = xr.DataArray(
#     final_tidal_surface,
#     coords={'y': y_target, 'x': x_target},
#     dims=['y', 'x'],
#     attrs={'crs': dem_xr.rio.crs}
# )
# #%%

# # interpolate missing values
# tidal_surface = tidal_surface.interpolate_na(dim='x', method='linear')
# # tidal_surface = tidal_surface.interpolate_na(dim='y', method='linear')
# #%%
# # smooth the tidal surface
# from scipy.ndimage import gaussian_filter
# tidal_surfaceFilt = xr.DataArray(
#     gaussian_filter(tidal_surface, sigma=50, mode='reflect'), #<--- equivalent of 50*3m = 150m smoothing
#     coords={'y': y, 'x': x},
#     dims=['y', 'x'],
#     attrs={'crs': dem_xr.rio.crs}
# )
# # %%
# # %% TEST FIG: PLOT SUBSET of TIDAL SURFACE

# subset = tidal_surface
# extent = [subset.coords['x'].min(), subset.coords['x'].max(), 
#           subset.coords['y'].min(), subset.coords['y'].max()]


# plt.figure(figsize=(5,5))
# plt.imshow(subset, cmap='viridis',extent=extent)
# # flip the y axis
# plt.gca().invert_yaxis()
# plt.colorbar(label='Elevation (m)')
# plt.title('Tidal Surface (MHHW), in mNAVD88')


# plt.xlabel('Easting (m)')
# plt.ylabel('Northing (m)')


# #equal aspect ratio
# plt.gca().set_aspect('equal', adjustable='box')
# plt.show()


#%%
#  save the tidal surface as xarray

tidal_surfaceFilt = tidal_surfaceFilt.rio.write_crs(dem_xr.rio.crs)

#copy the crs from the dem_xr.rio.crs



tidal_surfaceFilt.attrs = dem_xr.attrs

#%%


# write the tidal surface to a geotiff
tidal_surfaceFilt.rio.write_nodata(np.nan)
tidal_surfaceFilt.rio.to_raster('./tidal_surface/tidal_surface.tif')

# remove coord spatial_ref before saving to netcdf
tidal_surfaceFilt = tidal_surfaceFilt.drop('spatial_ref')

tidal_surfaceFilt.to_netcdf('./tidal_surface/tidal_surface.nc')

#%% TEST: Get min and max of the tidal surface
# load the tidal tidal_surface
tidal_surface = xr.open_dataarray('./tidal_surface/tidal_surface.nc')
print(tidal_surface.min(), tidal_surface.max())
# %% TEST FIG: PLOT SUBSET of TIDAL SURFACE

subset = tidal_surface.isel(x=slice(-5000,-1), y=slice(0, 12000))
extent = [subset.coords['x'].min(), subset.coords['x'].max(), 
          subset.coords['y'].min(), subset.coords['y'].max()]


plt.figure(figsize=(5,5))
plt.imshow(subset, cmap='viridis',extent=extent)
# flip the y axis
plt.gca().invert_yaxis()
plt.colorbar(label='Elevation (m)')
plt.title('Tidal Surface (MHHW), in mNAVD88')


plt.xlabel('Easting (m)')
plt.ylabel('Northing (m)')


#equal aspect ratio
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
# %%

#%%
# generation normals to the dem_xr, and make transects from the shoreline
# %%
# make the transects
# get the normals to the shoreline
