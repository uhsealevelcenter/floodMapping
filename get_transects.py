#%%
import pandas as pd
from scipy.interpolate import griddata, RegularGridInterpolator
from get_flood_raster_KSC import calculate_flooding_days, threshold_to_days, load_dem
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from scipy.ndimage import label, generate_binary_structure, binary_dilation

#%%
NOAA_SLR_DEM_PATH = './inputData/NOAA_SLR_DEM/NOAA_SLR_DEM_J995345.tif'
dem_xr = load_dem(NOAA_SLR_DEM_PATH) #



#%%
# # find the shoreline by finding the last x location where the dem_xr is greater than 0 for each y
# # This is a klugey hack, but it works for now
shoreline = dem_xr.x.where(dem_xr>-50).max(dim='x')

shoreline_coords = np.column_stack((shoreline.values, shoreline.y.values))

# #%%

# transform = dem_xr.transform
dem_extent = [dem_xr.coords['x'].min(), dem_xr.coords['x'].max(), 
          dem_xr.coords['y'].min(), dem_xr.coords['y'].max()]
#%%
# Plot the shoreline
plt.imshow(dem_xr, extent=dem_extent, cmap='terrain')
plt.plot(shoreline_coords[:, 0], shoreline_coords[:, 1], 'b-', label='Shoreline')
plt.colorbar(label='Elevation (m)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Shoreline')
plt.legend()
plt.show()

# #%%
#%%
def generate_transects(normals, length=10, num_points=100):
    transects = []
    for mid_point, normal in normals:
        start = mid_point - normal * length
        end = mid_point #+ normal * length / 2
        x_transect = np.linspace(start[0], end[0], num_points)
        y_transect = np.linspace(start[1], end[1], num_points)
        transects.append((x_transect, y_transect))
    return transects

length = 1000  # Length of transects in meters
num_points = 2  # Number of points along each transect
transects = generate_transects(normals, length, num_points)
#%%
# Plot transects
plt.imshow(dem_xr.where(dem_xr>-5), extent=dem_extent, cmap='terrain')
plt.plot(shoreline_coords[:, 0], shoreline_coords[:, 1], 'b-', label='Shoreline')
for x_transect, y_transect in transects:
    plt.plot(x_transect, y_transect, 'r-')
plt.colorbar(label='Elevation (m)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Shore-Normal Transects')
plt.legend()
plt.show()
#
#%%
# Create a mask for the points to the right of the inner transect
# draw a line corresponding to inner transect
#plot a line using points from the inner transect
first_points = [(x[0], y[0]) for x, y in transects]
x_inner, y_inner = zip(*first_points)
# extend line to top and bottom of the dem
y_top = dem_xr.coords['y'].max()
y_bottom = dem_xr.coords['y'].min()

x_top = x_inner[0]
x_bottom = x_inner[-1]

# add x,y top and bottom to the inner transect
x_inner = np.concatenate(([x_top], x_inner, [x_bottom]))
y_inner = np.concatenate(([y_top], y_inner, [y_bottom]))

#%%
# Create an empty mask with the same shape as your DEM values
mask = np.zeros_like(dem_xr.values, dtype=bool)

from scipy.interpolate import interp1d
# Create the interpolation function
interp_func = interp1d(y_inner, x_inner, kind='linear')

# Use the function to get interpolated and extrapolated x values
x_inner_interp = interp_func(dem_xr.y.values)


# for every row in y, set all x values greater than the interpolated x values to True
for i, y in enumerate(dem_xr.y.values):
    mask[i, dem_xr.x.values > x_inner_interp[i]] = True

#%%
# Plot the mask
plt.imshow(mask, extent=dem_extent, cmap='viridis')

# plt.imshow(dem_xr.where(dem_xr>-5), extent=dem_extent, cmap='terrain')
plt.plot(shoreline_coords[:, 0], shoreline_coords[:, 1], 'b-', label='Shoreline')
plt.plot(x_inner, y_inner, 'r-', label='Inner Transect')

# Plot the mask
# plt.imshow(mask, extent=dem_extent, cmap='viridis')
# %%
# save the mask
mask_3000_inland_xr = xr.DataArray(mask, coords={'y': dem_xr.y, 'x': dem_xr.x}, dims=['y', 'x'])
# %%
# Save the mask to a NetCDF file
mask_3000_inland_xr.to_netcdf('./mask_3000_inland.nc')
# %%
