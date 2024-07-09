#%%
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import xarray as xr
import numpy as np
import os
import matplotlib.colors as mcolors
#%%
# Define custom colormap with specific colors for each category
cmap = mcolors.ListedColormap(['green', 'magenta', 'cyan', 'blue'])
boundaries = [-0.5, 0.5, 1.5, 2.5, 3.5]
norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=True)
#%%
# Define the directory containing the NetCDF files
output_dir = './connected_masks'

# Get a list of all NetCDF files in the output directory
netcdf_files = [f for f in os.listdir(output_dir) if f.endswith('.nc')]

# Function to extract the numeric part from the filename
def extract_threshold(filename):
    # Extract the part of the filename between the last underscore and 'm.nc'
    threshold_str = filename.split('_')[-1].replace('mMHHW.nc', '')
    return float(threshold_str)

# Sort the files based on the extracted numeric values
netcdf_files = sorted(netcdf_files, key=extract_threshold)

# Load a sample file to get the extent and set up the plot
sample_file = xr.open_dataset(os.path.join(output_dir, netcdf_files[0]))
extent = [sample_file.coords['x'].min(), sample_file.coords['x'].max(), 
          sample_file.coords['y'].min(), sample_file.coords['y'].max()]

# Create the figure and axis for the plot
fig, ax = plt.subplots(figsize=(8, 10))

# Function to initialize the plot
def init():
    img.set_data(np.zeros((len(sample_file.y), len(sample_file.x))))
    return [img]

# Function to update the plot for each frame
def update(frame):
    print(f'Processing frame {frame} of {len(netcdf_files)}')
    file_path = os.path.join(output_dir, netcdf_files[frame])
    ds = xr.open_dataset(file_path)
    mask_combined_xr = ds['mask_combined']
    img.set_array(mask_combined_xr)
    elevation_threshold = netcdf_files[frame].split('_')[-1].replace('mMHHW.nc', '')
    ax.set_title(f'Connectedness Mask: {elevation_threshold}m MHHW')
    return [img]

# Add the initial image to the plot
img = ax.imshow(np.zeros((len(sample_file.y), len(sample_file.x))), cmap=cmap, norm=norm, extent=extent)

# Add a star for the ocean point
ocean_pt = (0, len(sample_file.x) - 1)
inland_pt = (11100, 5500)

ocean_pt = (sample_file.y.max(), sample_file.x.max())
inland_pt = (3150000,537000) #this should be the point in the inland lagoon
north_inland_pt = (3175000, 530000) #this should be the point in the inland lagoon
west_inland_pt = (3160000, 525000) #this should be the point in the inland lagoon


ax.scatter(ocean_pt[1]-200, ocean_pt[0]-200, marker='*', color='k', s=100)
ax.text(ocean_pt[1]-1000, ocean_pt[0]-1000, 'Ocean\nStart\nPoint', color='k', fontsize=10, verticalalignment='top', horizontalalignment='right')

# Add a star for the inland point
ax.scatter(inland_pt[1], inland_pt[0], marker='*', color='k', s=100)
ax.text(inland_pt[1], inland_pt[0]-1000, 'Inland\nStart\nPoint', color='k', fontsize=10, verticalalignment='top', horizontalalignment='center')

# Add colorbar
cbar = plt.colorbar(img, ticks=[0, 1, 2, 3])
cbar.set_ticklabels(['No Flood', 'Disconnected\nLow-lying', 'Lagoon-connected', 'Coastal-connected'], rotation=90, verticalalignment='center')

# Add labels
plt.xlabel('Easting (m)')
plt.ylabel('Northing (m)')

update(frame=0)
#%%
# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(netcdf_files), init_func=init, blit=True)
# ani = animation.FuncAnimation(fig, update, frames=50, init_funtc=init, blit=True)

# Save the animation as a GIF
ani.save('connectedness_mask_animation_MHHW.gif', writer='pillow', fps=10)

plt.show()
# %%
