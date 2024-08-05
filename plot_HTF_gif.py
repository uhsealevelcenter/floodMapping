#%%
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import xarray as xr
import numpy as np
import os
import matplotlib.colors as mcolors
import geopandas as gpd
from shapely.geometry import box
import contextily as ctx
import pandas as pd
#%%
# Define custom colormap with specific colors for each category
# cmap = mcolors.ListedColormap(['green', 'orange','magenta', 'cyan', 'blue'])
# boundaries = [-99, 0, 20, 50, 100, 200, 365]
# norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=True)
#%%
# Define the directory containing the Tif files
raster_dir = './flood_days_raster'
scenario = 'int'
output_dir = os.path.join(raster_dir, scenario)

# Get a list of all Tif files in the output directory
tif_files = [f for f in os.listdir(output_dir) if f.endswith('.tif')]
#%%
# Function to extract the numeric part from the filename
def extract_year(filename):
    # Extract the part of the filename between the last underscore and 'm.nc'
    year_str = filename.split('_')[0]
    return float(year_str)


# Sort the files based on the extracted numeric values
tif_files = sorted(tif_files, key=extract_year)
#%%
# Load a sample file to get the extent and set up the plot
ds = xr.open_dataarray(os.path.join(output_dir, tif_files[0]))[0]

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
#%%
# Create the figure and axis for the plot
fig, ax = plt.subplots(figsize=(8, 10))
# #%%


# Mask the zeros for transparency
masked_array = np.ma.masked_where(ds <= 0, ds)
# Plot the data with the correct extent for the data in EPSG:26917
img = ax.imshow(masked_array[::10,::10], extent=[min_x, max_x, min_y, max_y],zorder=2,alpha=1)
img.set_cmap('plasma')

# Add the basemap (ESRI Satellite)
ctx.add_basemap(ax, crs='EPSG:26917', source=ctx.providers.Esri.WorldImagery, zoom=11, attribution_size=6)


yrtext = ax.text(0.95, 0.95, tif_files[0][0:4], verticalalignment='top', horizontalalignment='right', transform=ax.transAxes, color='white', fontsize=24, fontweight='bold')

# Load locations of interest from KSC_locations.csv
locations = pd.read_csv('KSC_locations.csv')

# Ensure the Easting and Northing columns are numeric
locations['Easting'] = pd.to_numeric(locations['Easting'], errors='coerce')
locations['Northing'] = pd.to_numeric(locations['Northing'], errors='coerce')

# Plot the locations on the map as stars with labels
for i, location in locations.iterrows():
    ax.plot(location['Easting'], location['Northing'], marker='*', color='cyan', markersize=20, markeredgecolor='black', markeredgewidth=1.5,zorder=3)
    location_name_multiline = '\n'.join(location['Name'].split())
    # limit to only 2 lines
    location_name_multiline = '\n'.join(location_name_multiline.split('\n')[:2])
    ax.text(location['Easting']+1000, location['Northing'], location_name_multiline, fontsize=14, color='cyan', zorder=3)



# Function to initialize the plot
def init():
    img.set_data(np.zeros((len(sample_file.y), len(sample_file.x))))
    return [img]

# Function to update the plot for each frame
def update(frame):
    print(f'Processing frame {frame} of {len(tif_files)}')
    file_path = os.path.join(output_dir, tif_files[frame])
    ds = xr.open_dataarray(file_path)[0]
    # Mask the zeros for transparency
    masked_array = np.ma.masked_where(ds <= 0, ds)  
    img.set_array(masked_array[::10,::10])
    year_str = tif_files[frame].split('_')[0]
    # add text in top right for the year_str
    yrtext.set_text(tif_files[frame][0:4])
    return [img]

# # Add colorbar
cbar = plt.colorbar(img)
cbar.set_label('High Tide Flooding Days per Year')

ax.set_title(f'SLR scenario: {scenario}')

# Add labels
plt.xlabel('Easting (m)')
plt.ylabel('Northing (m)')


update(frame=5)
#%%
# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(tif_files), init_func=init, blit=True)
# ani = animation.FuncAnimation(fig, update, frames=50, init_funtc=init, blit=True)

# Save the animation as a GIF
savename = f'htf_{scenario}_animation_MHHW.gif'
ani.save(savename, writer='pillow', fps=2)

plt.show()
# %%
