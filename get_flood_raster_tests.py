#%%
from get_flood_raster_KSC_2gauge import calculate_flooding_days, threshold_to_days, load_dem, calculate_flooding_days_with_mask
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs

#**********************TESTING AREA********************************
    #%%
#TEST 1: plot the flooding days by threshold
#plot totals by threshold, with x-axis as threshold and y-axis as total days flooded

scenario = 'int_low'
df_inland = threshold_to_days(scenario, TG_SOURCE='./inputData/02248380_MHHW')
df_coast = threshold_to_days(scenario, TG_SOURCE='./inputData/8721604_MHHW')

plt.plot(df_inland.loc[2100])
plt.plot(df_coast.loc[2100])
plt.xlabel('Threshold (m)')
plt.ylabel('Total Days Flooded')
# only look at 0-1m for now
plt.xlim(0, 1)

# make a vertical line for the 2ft mark
plt.axvline(0.61, color='r', linestyle='--')

# add a legend
plt.legend(['Total Days Flooded Inland','Total Days Flooded Coast', '2ft Threshold'])

# add a title
plt.title(f'Total Days Flooded by Threshold for {scenario} Scenario')

# save to viz folder as png
plt.savefig(f'./viz/{scenario}_threshold_to_days.png')

#%%
## TEST 2: make a dummy dem_xr to test
dem_xr = xr.DataArray(
    np.array([[6.481, 0.239, 0.614, 0.832], [0.68, 0.45, 0.61, 0.37], [0.48, 0.2, 0, 0.4], [0.48, 0.2, 0.61, 0.6]]),
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
# calculate_flooding_days(dem_xr, mhhw_xr_aligned, scenario, year, TG_SOURCE)
flooding_days,elevations = calculate_flooding_days(dem_xr, mhhw_xr, scenario, year, TG_SOURCE='./inputData/02248380_MHHW')

#plot the flooding days
plt.imshow(flooding_days)
plt.colorbar()
#%% TEST 3: 
NOAA_SLR_DEM_PATH = './inputData/NOAA_SLR_DEM/NOAA_SLR_DEM_J995345.tif'
TIDAL_SURFACE_PATH = './tidal_surface/tidal_surface.tif'
dem_xr = load_dem(NOAA_SLR_DEM_PATH)
mhhw_xr = load_dem(TIDAL_SURFACE_PATH)
mhhw_xr_aligned = mhhw_xr.rio.reproject_match(dem_xr)

scenario = 'int'
year = 2100
scenario_dir = f'./flood_days_raster/{scenario}'

# flooding_days = calculate_flooding_days(dem_xr, mhhw_xr_aligned, scenario, year, TG_SOURCE='./inputData/02248380')

flooding_days_inland,elevations = calculate_flooding_days(dem_xr, mhhw_xr_aligned, scenario, year, TG_SOURCE='./inputData/02248380_MHHW')
flooding_days_coast, elevations = calculate_flooding_days(dem_xr, mhhw_xr_aligned, scenario, year, TG_SOURCE='./inputData/8721604_MHHW')

# make a mask, where if mhhw_xr_aligned <0.1, then use flooding_days_inland, else use flooding_days_coast
mask = np.full_like(mhhw_xr_aligned, 0)
mask[mhhw_xr_aligned < 0.1] = 1
mask_xr = xr.DataArray(mask, dims=['y', 'x'])

flooding_days_masked_xr = xr.DataArray(np.full_like(flooding_days_inland, np.nan), dims=['y', 'x'])

flooding_days_masked_xr = flooding_days_masked_xr.where(mask_xr ==1, flooding_days_inland)
flooding_days_masked_xr = flooding_days_masked_xr.where(mask_xr == 0, flooding_days_coast)

os.makedirs(scenario_dir, exist_ok=True)

# flooding_days.rio.to_raster(os.path.join(scenario_dir, f'{year}.tif'))

#open this raster in QGIS to check stuff, OR:
#%%
#plot the flooding days
plt.imshow(flooding_days_masked_xr)
plt.colorbar()

# zoom in to x=4000:5000, y=4000:5000
# plt.xlim(4800, 4900)
# plt.ylim(4900, 5000)

# set the colorbar to be from -2 to 2
plt.clim(-1, 365)

# %%
# find all elevations where flooding_days_inland is 0
zerodays = elevations[flooding_days_coast == -50]

# plot the histogram of elevations where flooding_days_inland is -50, with bins from 0 to 1 in 0.01 increments
plt.hist(zerodays, bins=np.arange(-2, 4, 0.01))
plt.hist(elevations[flooding_days_inland == -50], bins=np.arange(-2, 4, 0.01))


# %%
# TEST 4:
# Make a mask, where we get flooding days from inland gauge where tidal surface is classified as non-tidal
# and get flooding days from coastal gauge where tidal surface is classified as tidal

# make a dummy tidal surface
tidal_surface = xr.DataArray(
    np.array([[0, 1, 1, 0], [.98632, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1]]),
    coords={'y': np.arange(4), 'x': np.arange(4)},
    dims=['y', 'x'],
    attrs={'crs': 'EPSG:26917'}
)

# make a mask
mask = np.full_like(tidal_surface, 0)
mask[tidal_surface == 0] = 1

# make a dummy dem_xr
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
flooding_days_inland = calculate_flooding_days(dem_xr, mhhw_xr, scenario, year, TG_SOURCE='./inputData/02248380')
flooding_days_coast = calculate_flooding_days(dem_xr, mhhw_xr, scenario, year, TG_SOURCE='./inputData/TGanalysis')

# apply the mask
flooding_days_masked = np.full_like(flooding_days_inland, np.nan)
flooding_days_masked[mask == 0] = flooding_days_inland[mask == 0]
flooding_days_masked[mask == 1] = flooding_days_coast[mask == 1]

#plot the flooding days
plt.imshow(flooding_days_masked)
plt.colorbar()

# %%
#plot the tidal surface (from ./tidal_surface/tidal_surface.tif)
NOAA_SLR_DEM_PATH = './inputData/NOAA_SLR_DEM/NOAA_SLR_DEM_J995345.tif'
TIDAL_SURFACE_PATH = './tidal_surface/tidal_surface.tif'
dem_xr = load_dem(NOAA_SLR_DEM_PATH)
mhhw_xr = load_dem(TIDAL_SURFACE_PATH)
mhhw_xr_aligned = mhhw_xr.rio.reproject_match(dem_xr)
#%%
dem_xr = dem_xr.isel(y=slice(300, dem_xr.shape[0]), x=slice(0,dem_xr.shape[1]-300))
mhhw_xr_aligned = mhhw_xr_aligned.isel(y=slice(300, mhhw_xr_aligned.shape[0]), x=slice(0,mhhw_xr_aligned.shape[1]-300))
#%%
scenario = 'int_low'
year = 2100
flooding_days, elevations = calculate_flooding_days_with_mask(dem_xr, mhhw_xr_aligned, scenario, year)
#%%
# plot the flooding days
flooding_days[::50,::50].where(flooding_days>0).plot()
scenario_dir = os.path.join(FLOOD_DAYS_RASTER_DIR, scenario)
os.makedirs(scenario_dir, exist_ok=True)

flooding_days.rio.to_raster(os.path.join(scenario_dir, f'{year}_{scenario}_per50.tif'))




#%%
boundaries = [-0.5, 0.5, 1.5, 2.5,3.5]
#make cmap from seaborn with 4 colors
cmap = mpl.colormaps['tab20']
norm = BoundaryNorm(boundaries, cmap.N, clip=True)

extent = [mask_combined_xr.coords['x'].min(), mask_combined_xr.coords['x'].max(), 
          mask_combined_xr.coords['y'].min(), mask_combined_xr.coords['y'].max()]

# make a figure with coordinate reference system (crs) of the dem_xr
# fig,ax = plt.subplots(figsize=(8,3), subplot_kw={'projection': ccrs.UTM(17)})
fig, ax = plt.subplots(figsize=(15, 10))

img = plt.imshow(mask_combined_xr, cmap=cmap, norm=norm, extent=extent)
# mask_combined_xr.sel(x=slice(525000,550000), y=slice(3170000,3140000)).plot(ax = ax, cmap=cmap, norm=norm)
# .sel(x=slice(525000,540000), y=slice(3160000,3140000))
# show the x and y labels
# ax.gridlines(draw_labels=True)

# Add a star for the ocean point
ax.scatter(dem_xr['x'][ocean_pt[1]], dem_xr['y'][ocean_pt[0]], marker='*', color='k', s=100)
# ax.scatter(ocean_pt[1], ocean_pt[0], marker='*', color='b', s=100)
ax.text(dem_xr['x'][ocean_pt[1]]-500, dem_xr['y'][ocean_pt[0]]-1000, 'Ocean\nStart\nPoint', color='k', fontsize=10, verticalalignment='top', horizontalalignment='right')

# Add a star for the inland point
ax.scatter(dem_xr['x'][inland_pt[1]], dem_xr['y'][inland_pt[0]], marker='*', color='k', s=100)
# add label with star
ax.text(dem_xr['x'][inland_pt[1]], dem_xr['y'][inland_pt[0]]+1000, 'Inland\nStart\nPoint', color='k', fontsize=10, verticalalignment='bottom', horizontalalignment='center')

# ax.scatter(inland_pt[1], inland_pt[0], marker='*', color='b', s=100)
# Add a star for the IRNorth point
# plt.scatter(IRNorth_pt[1], IRNorth_pt[0], marker='*', color='g', s=100)

# Add colorbar
# cbar = plt.colorbar(img, ticks=[0.5, 1.5, 2.5, 3.5])
cbar = plt.colorbar(img, ticks=[0, 1, 2, 3])

cbar.set_ticklabels(['No Flood', 'Disconnected', 'Inland-connected', 'Coastal-connected'], rotation=90, verticalalignment='center')
# cbar.set_ticklabels(['No Flood', 'Inland-connected', 'Disconnected'], rotation=90, verticalalignment='center')

# add title with depth 
title_str = f'Connectedness Mask: {elevation_threshold}m NAVD88' 
plt.title(title_str)

# add x and y labels
plt.xlabel('Easting (m)')
plt.ylabel('Northing (m)')

plt.show()


#%%
cmap = mpl.colormaps['tab20']



plt.imshow(mask_combined_xr)
#flip the y-axis
# plt.gca().invert_yaxis()

# put a star for the ocean point
plt.scatter(ocean_pt[1], ocean_pt[0], marker='*', color='r',s=100)
plt.colorbar()
plt.clim(0,1)

# add discrete colorbar with distinct colors for each label


#zoom in to x=5800:8000, y=10000:12000
# plt.xlim(5400, 5700)
# plt.ylim(11000, 11300)


#%%
#what's the value of dem_xr along the top edge?
plt.plot(mask_to_apply[-1,:])
# %%
#show histogram of mask_coast above 0.1m
# plt.hist(mask_to_apply_inland)

#%%

import contextily as ctx
import geopandas as gpd
from shapely.geometry import box
from matplotlib.patches import Polygon,Rectangle
# Create the figure and axis for the plot
fig, ax = plt.subplots(figsize=(15,20))

# Assuming your data is already in EPSG:26917
x_coords = dem_xr.coords['x'].values
y_coords = dem_xr.coords['y'].values

# Define the bounding box for the data
min_x, max_x = x_coords.min(), x_coords.max()
min_y, max_y = y_coords.min(), y_coords.max()

cmap = plt.get_cmap('jet')

# Mask the zeros for transparency
masked_fd = np.ma.masked_where(mask <= 150, mask)

# Get the min and max values of masked_fd
vmin, vmax = np.min(masked_fd), np.max(masked_fd)

# Plot the data with the correct extent for the data in EPSG:26917
img = ax.imshow(masked_fd , cmap = cmap, extent=[min_x, max_x, min_y, max_y],zorder=2,alpha=1, vmin=vmin, vmax=vmax)

# Add the basemap (ESRI Satellite)
# ctx.add_basemap(ax, crs='EPSG:26917', source=ctx.providers.Esri.WorldImagery, zoom=12, attribution_size=6,zorder=1)

# # Function to add zebra stripes to the border
# def add_zebra_frame(ax, min_x, max_x, min_y, max_y, lw=5, segment_length=5000):
#     # Calculate the starting positions aligned with segment_length
#     start_x = min_x - (min_x % segment_length)
#     start_y = min_y - (min_y % segment_length)
    
#     num_segments_x = int(np.ceil((max_x - start_x) / segment_length))
#     num_segments_y = int(np.ceil((max_y - start_y) / segment_length))

#     # Draw horizontal stripes at the top and bottom
#     for i in range(num_segments_x):
#         color = 'black' if i % 2 == 0 else 'white'
#         stripe_x_start = start_x + i * segment_length
#         stripe_x_end = stripe_x_start + segment_length
#         ax.hlines([min_y + lw, max_y - lw], stripe_x_start, stripe_x_end, colors=color, linewidth=lw, zorder=4)

#     # Draw vertical stripes on the left and right
#     for j in range(num_segments_y):
#         color = 'black' if j % 2 == 0 else 'white'
#         stripe_y_start = start_y + j * segment_length
#         stripe_y_end = stripe_y_start + segment_length
#         ax.vlines([min_x + lw, max_x - lw], stripe_y_start, stripe_y_end, colors=color, linewidth=lw, zorder=4)

# # Add zebra stripes to the border
# add_zebra_frame(ax, min_x, max_x, min_y, max_y)

# # Add a solid black border around the plot
# rect = Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, linewidth=1, edgecolor='black', facecolor='none', zorder=4)
# ax.add_patch(rect)

# Add colorbar with the colormap that has alpha=1
cbar = plt.colorbar(mappable=ax.imshow(masked_fd ,cmap = cmap,  extent=[min_x, max_x, min_y, max_y], alpha=1, zorder=1, visible=False, vmin=vmin, vmax=vmax), ax=ax)# cbar.set_ticklabels(['Disconnected', 'Lagoon-connected', 'Coastal-connected'], rotation=90, verticalalignment='center')
cbar.set_label('Flooded Days')

# Add labels
plt.xlabel('Easting (m)')
plt.ylabel('Northing (m)')

# # Add title to the plot
# title_str = 'Connectedness Mask: '+ str(threshold) +'m NAVD88'
# plt.title(title_str)

# save_path = 'connectedness_mask_'+ str(threshold) + 'm_NAVD88.png'

# # Save the plot as a PNG
# plt.savefig(save_path, dpi=300, bbox_inches='tight')



plt.show()
# %%
dem_xr_lock.plot()
# %%

# go throught all elevations to check sizes of x,y
for elevation in np.arange(0.95,3, 0.01):
    mask = xr.open_dataarray(f'./connected_masks/mask_combined_{elevation:.2f}mMHHW.nc')
    print(mask.shape)


# %%
