# %%
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import os
import matplotlib.colors as mcolors
import contextily as ctx
import geopandas as gpd
from shapely.geometry import box
from matplotlib.patches import Polygon,Rectangle

# %%
# Define custom colormap with specific colors for each category
cmap = mcolors.ListedColormap(['magenta', 'cyan', 'blue'])
boundaries = [0.5, 1.5, 2.5, 3.5]
norm = mcolors.BoundaryNorm(boundaries, cmap.N)

cmap_alpha1 = mcolors.ListedColormap([(1,0,1, 1),  # magenta
                                      (0, 1, 1, 1),                  # cyan
                                      (0, 0, 1, 1)])                 # blue

# %%
# Define the directory containing the NetCDF files
output_dir = '.'
viz_dir = './viz'

# Get the specific NetCDF file ending with 0.88.nc
threshold = 0.09
netcdf_file = './connected_masks/mask_combined_'+ str(threshold) + 'mMHHW.nc'

# Load the file to get the extent and set up the plot
file_path = os.path.join(output_dir, netcdf_file)
ds = xr.open_dataset(file_path)
mask_combined_xr = ds['mask_combined']

# Assuming your data is already in EPSG:26917
x_coords = ds.coords['x'].values
y_coords = ds.coords['y'].values

# Define the bounding box for the data
min_x, max_x = x_coords.min(), x_coords.max()
min_y, max_y = y_coords.min(), y_coords.max()

# Create a GeoDataFrame with the bounding box to set the extent
bbox = gpd.GeoDataFrame({'geometry': [box(min_x, min_y, max_x, max_y)]}, crs="EPSG:26917")

# Print the extents to verify they are reasonable
print(f"Extent in EPSG:26917 - min_x: {min_x}, min_y: {min_y}, max_x: {max_x}, max_y: {max_y}")

# Mask the zeros for transparency
masked_array = np.ma.masked_where(mask_combined_xr == 0, mask_combined_xr)

# Debugging: Print masked array stats
print(f"Masked array min: {masked_array.min()}, max: {masked_array.max()}, mean: {masked_array.mean()}")
#%%
# Create the figure and axis for the plot
fig, ax = plt.subplots(figsize=(8, 10))



# Plot the data with the correct extent for the data in EPSG:26917
img = ax.imshow(masked_array, cmap=cmap, norm=norm, extent=[min_x, max_x, min_y, max_y],zorder=2,alpha=0.8)



# Add the basemap (ESRI Satellite)
ctx.add_basemap(ax, crs='EPSG:26917', source=ctx.providers.Esri.WorldImagery, zoom=12, attribution_size=6)

# Function to add zebra stripes to the border
def add_zebra_frame(ax, min_x, max_x, min_y, max_y, lw=5, segment_length=5000):
    # Calculate the starting positions aligned with segment_length
    start_x = min_x - (min_x % segment_length)
    start_y = min_y - (min_y % segment_length)
    
    num_segments_x = int(np.ceil((max_x - start_x) / segment_length))
    num_segments_y = int(np.ceil((max_y - start_y) / segment_length))

    # Draw horizontal stripes at the top and bottom
    for i in range(num_segments_x):
        color = 'black' if i % 2 == 0 else 'white'
        stripe_x_start = start_x + i * segment_length
        stripe_x_end = stripe_x_start + segment_length
        ax.hlines([min_y + lw, max_y - lw], stripe_x_start, stripe_x_end, colors=color, linewidth=lw, zorder=4)

    # Draw vertical stripes on the left and right
    for j in range(num_segments_y):
        color = 'black' if j % 2 == 0 else 'white'
        stripe_y_start = start_y + j * segment_length
        stripe_y_end = stripe_y_start + segment_length
        ax.vlines([min_x + lw, max_x - lw], stripe_y_start, stripe_y_end, colors=color, linewidth=lw, zorder=4)

# Add zebra stripes to the border
add_zebra_frame(ax, min_x, max_x, min_y, max_y)

# Add a solid black border around the plot
rect = Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, linewidth=1, edgecolor='black', facecolor='none', zorder=4)
ax.add_patch(rect)

# Add a star for the ocean point
ocean_pt = (0, len(x_coords) - 1)
inland_pt = (11100, 5500)

ax.scatter(x_coords[ocean_pt[1]], y_coords[ocean_pt[0]], marker='*', color='k', s=100, zorder=3)
ax.text(x_coords[ocean_pt[1]]-500, y_coords[ocean_pt[0]]-1000, 'Ocean\nStart\nPoint', color='w', fontsize=10, verticalalignment='top', horizontalalignment='right')

# Add a star for the inland point
ax.scatter(x_coords[inland_pt[1]], y_coords[inland_pt[0]], marker='*', color='k', s=100, zorder=3)
ax.text(x_coords[inland_pt[1]], y_coords[inland_pt[0]]+1000, 'Inland\nStart\nPoint', color='w', fontsize=10, verticalalignment='bottom', horizontalalignment='center')

# Add colorbar with the colormap that has alpha=1
cbar = plt.colorbar(mappable=ax.imshow(masked_array, cmap=cmap_alpha1, norm=norm, extent=[min_x, max_x, min_y, max_y], alpha=1, zorder=1, visible=False), ax=ax, ticks=[1, 2, 3])
cbar.set_ticklabels(['Disconnected', 'Lagoon-connected', 'Coastal-connected'], rotation=90, verticalalignment='center')

# Add labels
plt.xlabel('Easting (m)')
plt.ylabel('Northing (m)')

# Add title to the plot
title_str = 'Connectedness Mask: '+ str(threshold) +'m MHHW'
plt.title(title_str)

save_name = 'connectedness_mask_'+ str(threshold) + 'm_MHHW.png'

save_path = os.path.join(viz_dir, save_name)

# Save the plot as a PNG
plt.savefig(save_path, dpi=300, bbox_inches='tight')



plt.show()
# %%
# save mask_combined_xr to a geotiff
mask_combined_xr.rio.to_raster('mask_combined_' + str(threshold)+ 'mMHHW.tif')
# %%
