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
# Load ./flood_days_raster/int/2100.tif
# Load the raster data
file_path = './flood_days_raster/high/2020_high_per50.tif'

#open the tif file
ds = xr.open_dataarray(file_path)[0]

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
masked_array = np.ma.masked_where(ds <= 0, ds)

# Debugging: Print masked array stats
print(f"Masked array min: {masked_array.min()}, max: {masked_array.max()}, mean: {masked_array.mean()}")
#%%
# Create the figure and axis for the plot
fig, ax = plt.subplots(figsize=(8, 10))



# Plot the data with the correct extent for the data in EPSG:26917
img = ax.imshow(masked_array[::10,::10], extent=[min_x, max_x, min_y, max_y],zorder=2,alpha=0.8)



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
cbar = plt.colorbar(mappable=ax.imshow(masked_array, extent=[min_x, max_x, min_y, max_y], alpha=1, zorder=1, visible=False), ax=ax)
# cbar.set_ticklabels(['Disconnected', 'Lagoon-connected', 'Coastal-connected'], rotation=90, verticalalignment='center')

# Add labels
plt.xlabel('Easting (m)')
plt.ylabel('Northing (m)')

# # Add title to the plot
# title_str = 'Connectedness Mask: '+ str(threshold) +'m NAVD88'
plt.title(title_str)

# save_path = 'connectedness_mask_'+ str(threshold) + 'm_NAVD88.png'

# # Save the plot as a PNG
plt.savefig(save_path, dpi=300, bbox_inches='tight')



plt.show()
# %%
