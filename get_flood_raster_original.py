#%%
import rasterio
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import json
import cartopy.crs as ccrs
import rioxarray
#%%
# Path to the geotiff file
file_path = './NOAA_SLR_DEM/NOAA_SLR_DEM_J995345.tif'
trident_mhhw = 	0.336 # meters MHHW, I looked this up from NOAA datum info

def load_dem(file_path):
    """Load DEM data and return as an xarray DataArray with spatial dimensions and CRS."""
    with rasterio.open(file_path) as src:
        dem_data = src.read(1) - trident_mhhw  # Adjust elevation to MHHW from NAVD
        attrs = {
            'crs': src.crs,
            'transform': src.transform,
            'units': 'meters MHHW',
            'description': 'NOAA Sea Level Rise DEM, adjusted to MHHW',
        }
        dem_xr = xr.DataArray(
            dem_data, 
            coords={'y': np.linspace(src.bounds.top, src.bounds.bottom, src.height),
                    'x': np.linspace(src.bounds.left, src.bounds.right, src.width)},
            dims=['y', 'x'],
            attrs=attrs
        ).rio.write_crs(src.crs.to_string())
    return dem_xr
#%%
def calculate_flooding_days(dem_xr, scenario, year):
    """Calculate flooding days for a given year and scenario."""
    totals_by_threshold = {}
    for threshold in np.arange(0, 3.01, 0.01): # Thresholds from 0 to 3 meters
        with open(f'./TGanalysis/{scenario}/{int(threshold * 100):03d}.json') as f: # Load data from json files
            data = json.load(f)
        totals_by_threshold[threshold] = sum(data['monthly_percentiles'][str(year)]['50']) #only 50th percentile for now
    
    elevations = np.round(dem_xr.values, 2) # Round elevation values
    days_per_year = np.full_like(elevations,np.nan) # Initialize array to store flooding days
    for elevation, days in totals_by_threshold.items(): # Assign flooding days to elevation values
        days_per_year[elevations == elevation] = days 
    
    days_per_year[elevations <=0] = -10 # denote land under MHHW

    #if elevation is less than 0-mhhw, then it currently MHHW
    days_per_year[elevations <= 0-trident_mhhw] = -99 # Denote ocean
    
    # set nodata value to -999
    days_per_year[np.isnan(elevations)] = -999

    return xr.DataArray( # Return flooding days as an xarray DataArray
        days_per_year, 
        coords=dem_xr.coords, 
        dims=dem_xr.dims, 
        attrs={'crs': dem_xr.rio.crs, 'scenario': scenario, 'year': year, 'notes': 'Land under MHHW denoted by -10, Ocean denoted by -99'}
    )
#%%
def main():
    dem_xr = load_dem('./NOAA_SLR_DEM/NOAA_SLR_DEM_J995345.tif')
    scenarios = ['high', 'int', 'int_high', 'int_low', 'low']
    years = np.arange(2020, 2101, 10) # only by decade for now
    
    for scenario in scenarios:
        os.makedirs(f'./flood_days_raster/{scenario}', exist_ok=True)
        for year in years:
            flooding_days = calculate_flooding_days(dem_xr, scenario, year)
            flooding_days.rio.to_raster(f'./flood_days_raster/{scenario}/{year}.tif')

if __name__ == '__main__':
    main()

# %%
# # TEST:
# dem_xr = load_dem('./NOAA_SLR_DEM/NOAA_SLR_DEM_J995345.tif')
# scenario = 'int_low'
# year = 2100
# flooding_days = calculate_flooding_days(dem_xr, scenario, year)


# flooding_days.rio.to_raster(f'./flood_days_raster/{scenario}/{year}.tif')
# %%
# MAKE CSV for mapping KSC Locations
KSC_launch39A = [538682, 3164664]
KSC_visitors_center = [531143.87, 3155058.75]
KSC_tidegauge = [539834, 3143243]

# make a csv with name, easting, northing
import csv
with open('KSC_locations.csv', mode='w') as file:
    writer = csv.writer(file)
    writer.writerow(['Name', 'Easting', 'Northing'])
    writer.writerow(['Launch Pad 39A', KSC_launch39A[0], KSC_launch39A[1]])
    writer.writerow(["Visitor's Center", KSC_visitors_center[0], KSC_visitors_center[1]])
    writer.writerow(['Trident Pier Tide Gauge', KSC_tidegauge[0], KSC_tidegauge[1]])
