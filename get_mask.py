#%% Import necessary libraries
from get_flood_raster_KSC_2gauge import calculate_flooding_days, threshold_to_days, load_dem
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
from scipy.ndimage import label, generate_binary_structure, binary_dilation
import os

#%% Function to create a mask for a single elevation threshold and save to NetCDF
def create_and_save_mask(dem_xr_mhhw, threshold, ocean_pt, inland_pt,north_inland_pt,west_inland_pt, structure, dilation_iterations, output_dir):
    threshold = threshold.round(2)
    print(f"Processing elevation threshold: {threshold}")
    
    binary_mask = dem_xr_mhhw.round(2) < threshold
    dilated_mask = binary_dilation(binary_mask, structure=structure, iterations=dilation_iterations)
    labeled_array, num_features = label(dilated_mask, structure=structure)
    coastal_label = labeled_array[ocean_pt]
    inland_label = labeled_array[inland_pt]
    north_inland_label = labeled_array[north_inland_pt]
    west_inland_label = labeled_array[west_inland_pt]
    mask_to_apply_coast = labeled_array == coastal_label
    mask_to_apply_inland = labeled_array == inland_label
    mask_to_apply_north_inland = labeled_array == north_inland_label
    mask_to_apply_west_inland = labeled_array == west_inland_label

    mask_disconnected = (dem_xr_mhhw <= threshold) & (~mask_to_apply_coast) & (~mask_to_apply_inland) & (~mask_to_apply_north_inland) & (~mask_to_apply_west_inland)
    mask_to_apply_inland = mask_to_apply_inland | mask_to_apply_north_inland | mask_to_apply_west_inland

    mask_combined = np.zeros_like(mask_disconnected, dtype=np.int8)
    mask_combined[mask_disconnected] = 1
    mask_combined[mask_to_apply_inland] = 2
    mask_combined[mask_to_apply_coast] = 3
    
    # Create a DataArray for the mask
    mask_combined_xr = xr.DataArray(
        mask_combined,
        coords={'y': dem_xr_mhhw.y, 'x': dem_xr_mhhw.x},
        dims=['y', 'x'],
        attrs=dem_xr_mhhw.attrs
    )
    
    # Set attributes for the DataArray
    mask_combined_xr.attrs['description'] = f'Mask for coastal-connected (3), inland-connected (2), and disconnected areas (1) at elevation threshold {threshold}m. All else is 0.'
    mask_combined_xr.attrs['units'] = '1=disconnected, 2=inland-connected, 3=coastal-connected, 0=none of the above'
    mask_combined_xr.attrs['crs'] = dem_xr_mhhw.rio.crs.to_string()
    mask_combined_xr.name = 'mask_combined'
    
    # Save to NetCDF
    output_file = os.path.join(output_dir, f'mask_combined_{threshold:.2f}mMHHW.nc')
    mask_combined_xr.to_netcdf(output_file)
    print(f"Saved to {output_file}")
    
    # Free up memory
    del binary_mask, dilated_mask, labeled_array, mask_to_apply_coast, mask_to_apply_inland, mask_disconnected, mask_combined, mask_combined_xr
#%%
# Load DEM and create a copy with modifications
NOAA_SLR_DEM_PATH = './inputData/NOAA_SLR_DEM/NOAA_SLR_DEM_J995345.tif'
TIDAL_SURFACE_PATH = './tidal_surface/tidal_surface.tif'
dem_xr = load_dem(NOAA_SLR_DEM_PATH)
dem_xr_lock = dem_xr.copy()

# Add elevated structure to the DEM to indicate the lock
dem_xr_lock[11225:11260, 5500:5505] = 2 # this number is aribtrary

# Make artificial opening to the ocean
# dem_xr_lock[0:10, 4000:10000] = np.min(dem_xr)

# Make artificial opening to the inland lagoon
dem_xr_lock[0:10, 1000:3000] = np.min(dem_xr)

# Create dem_xr in mhhw using ./tidal_surface/tidal_surface.tif
mhhw_xr = load_dem(TIDAL_SURFACE_PATH)
mhhw_xr_aligned = mhhw_xr.rio.reproject_match(dem_xr)

# Put the DEM in MHHW space
# subtract the MHHW from the altered DEM  - we are now working in MHHW space
dem_mhhw = dem_xr_lock.values - mhhw_xr_aligned.values
dem_mhhw = np.round(dem_mhhw, 2)

# Remove all areas by taking smaller slice

#%%
# mask dem_mhhw an xarray DataArray
dem_mhhw_xr = xr.DataArray(
    dem_mhhw,
    coords={'y': dem_xr.y, 'x': dem_xr.x},
    dims=['y', 'x'],
    attrs = {
            'transform': dem_xr.transform,
            'units': 'meters MHHW',
            'description': 'NOAA Sea Level Rise DEM, adjusted to MHHW',
        }
).rio.write_crs(dem_xr.rio.crs.to_string())
#%%
# remove 1000 rows from the top and right
dem_mhhw_xr = dem_mhhw_xr.isel(y=slice(300, dem_xr.shape[0]), x=slice(0,dem_xr.shape[1]-300))
#%%
# Define parameters
elevation_coords = np.arange(2.71, 3.8, 0.01)
structure = generate_binary_structure(2, 2)
dilation_iterations = 1
ocean_pt = (dem_mhhw_xr.y.max(), dem_mhhw_xr.x.max())
inland_pt = (3150000,537000) #this should be the point in the inland lagoon
north_inland_pt = (3175000, 530000) #this should be the point in the inland lagoon
west_inland_pt = (3160000, 525000) #this should be the point in the inland lagoon

# get indices of ocean and inland points
ocean_pt_index = (dem_mhhw_xr.y.values.tolist().index(ocean_pt[0]), dem_mhhw_xr.x.values.tolist().index(ocean_pt[1]))
# Get the indices nearest to the inland point
inland_pt_index = (np.abs(dem_mhhw_xr.y.values - inland_pt[0])).argmin(), (np.abs(dem_mhhw_xr.x.values - inland_pt[1])).argmin()
north_inland_pt_index = (np.abs(dem_mhhw_xr.y.values - north_inland_pt[0])).argmin(), (np.abs(dem_mhhw_xr.x.values - north_inland_pt[1])).argmin()
west_inland_pt_index = (np.abs(dem_mhhw_xr.y.values - west_inland_pt[0])).argmin(), (np.abs(dem_mhhw_xr.x.values - west_inland_pt[1])).argmin()

output_dir = './connected_masks'  # Define your output directory

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
#%%
# # Process and save masks for each elevation threshold
for threshold in elevation_coords:
    # mask = xr.open_dataarray(f'./connected_masks/mask_combined_{threshold:.2f}mMHHW.nc')
    # if mask.shape[0] != dem_mhhw_xr.shape[0] or mask.shape[1] != dem_mhhw_xr.shape[1]:
        create_and_save_mask(dem_mhhw_xr, threshold, ocean_pt_index, inland_pt_index, north_inland_pt_index,west_inland_pt_index,structure, dilation_iterations, output_dir)
# %%
# create_and_save_mask(dem_mhhw_xr, 0.85, ocean_pt_index, inland_pt_index, north_inland_pt_index,west_inland_pt_index,structure, dilation_iterations, output_dir)

# %%
#plot dem_mhhw_xr as sanity check

# plot a slice of the DEM
# Plot a slice of the DEM with specified figure size
# Corrected plotting command
plt.figure(figsize=(5, 5))

# plot every 1000 to spped up plotting
dem_xr[::10, ::10].where((dem_xr>0) & (dem_xr<6)).plot()# %%

# add pts
plt.scatter(ocean_pt[1], ocean_pt[0], color='r', s=100)
plt.scatter(inland_pt[1], inland_pt[0], color='b', s=100)
plt.scatter(north_inland_pt[1], north_inland_pt[0], color='g', s=100)
plt.scatter(west_inland_pt[1], west_inland_pt[0], color='y', s=100)

# %%
# plot ./connected_masks/mask_combined_1.00mMHHW.nc
mask_combined = xr.open_dataset('./connected_masks/mask_combined_-0.56mMHHW.nc')
mask_combined['mask_combined'][::10, ::10].plot()
# %%

# plot dem_xr, but only show where mask_combined == 0

# only show elevations from -1 to 0
dem_xr.where((mask_combined['mask_combined'] == 0) & (dem_xr > -1) & (dem_xr < -0.5))[::10,::10].plot()
# %%
