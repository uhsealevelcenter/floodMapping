#%%
# This script calculates the number of flooding days for different scenarios and years, 
# and is used to generate the raster files in the 'flood_days_raster' directory, 
# made for Kennedy Space Center (KSC) area.

# Import libraries
import rasterio
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import json
import cartopy.crs as ccrs
import rioxarray
import os
import pandas as pd

# Constants
NOAA_SLR_DEM_PATH = './inputData/NOAA_SLR_DEM/NOAA_SLR_DEM_J995345.tif'
TIDAL_SURFACE_PATH = './tidal_surface/tidal_surface.tif'
TRIDENT_MHHW = 0.336  # meters MHHW from NOAA datum info (https://tidesandcurrents.noaa.gov/datums.html?datum=NAVD88&units=1&epoch=0&id=8721604&name=Trident+Pier%2C+Port+Canaveral&state=FL)
SCENARIOS = ['high', 'int', 'int_high', 'int_low', 'low']
YEARS = np.arange(2020, 2101, 10)  # Decadal intervals
FLOOD_DAYS_RASTER_DIR = './flood_days_raster'
TGDATA_DIR_COAST = './inputData/TGanalysis'
TGDATA_DIR_INLAND = './inputData/02248380'
#%%

def load_dem(file_path=NOAA_SLR_DEM_PATH):
    """
    Load DEM data and return as an xarray DataArray with spatial dimensions and CRS.

    Parameters:
    - file_path: str, path to the DEM geotiff file.

    Returns:
    - dem_xr: xarray DataArray, DEM data adjusted to MHHW with spatial coordinates.
    """
    with rasterio.open(file_path) as src:
        dem_data = src.read(1)
        attrs = {
            'crs': src.crs,
            'transform': src.transform,
            'units': 'meters NAVD88',
            'description': 'NOAA Sea Level Rise DEM, NAVD88',
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
# Calculate tidal surface using vDatum
# https://vdatum.noaa.gov/docs/datums.html


#%%
def threshold_to_days(scenario):
    """
    Calculate the number of flooding days at different thresholds for a given scenario.

    Parameters:
    - scenario (str): The name of the scenario.

    Returns:
    - df (pandas.DataFrame): A DataFrame containing the number of flooding days at different thresholds.
    """

    totals_by_threshold = {}
    for threshold in np.arange(0, 3.01, 0.01):  # Thresholds from 0 to 3 meters
        with open(os.path.join(TGDATA_DIR, f'{scenario}/{int(threshold * 100):03d}.json')) as f:
            data = json.load(f)
        totals_by_threshold[threshold] = data['annual_percentiles']['percentiles']['50']

    # turn this into a dataframe
    df = pd.DataFrame(totals_by_threshold)
    df.index = np.arange(2020, 2101, 1)
    return df

#%%
def calculate_flooding_days(dem_xr, mhhw_xr_aligned, scenario, year):
    """
    Calculate the number of flooding days for a given Digital Elevation Model (DEM) dataset,
    scenario, and year.

    Parameters:
        dem_xr (xarray.DataArray): The Digital Elevation Model dataset adjusted to MHHW.
        scenario (str): The flood scenario.
        year (int): The year for which to calculate the flooding days.

    Returns:
        xarray.DataArray: A DataArray containing the number of flooding days for each elevation
        in the DEM dataset, with special values assigned for land under MHHW, ocean, and no data.

    """
    df = threshold_to_days(scenario)

    # subtract the MHHW from the DEM  - we are now working in MHHW space
    elevations = dem_xr.values - mhhw_xr_aligned.values
    elevations = np.round(elevations, 2)

    days_per_year = np.full_like(elevations, np.nan)
    for elevation, days in df.loc[year].items():
        days_per_year[elevations == elevation] = days

    #set nans in days_per_year to 0
    days_per_year[np.isnan(days_per_year)] = 0    

    # Assign special values for land under MHHW and ocean
    days_per_year[elevations<=0] = -1  # Land under MHHW
    days_per_year[dem_xr.values<=-99] = -99  # No data

    

    return xr.DataArray(
        days_per_year,
        coords=dem_xr.coords,
        dims=dem_xr.dims,
        attrs={'crs': dem_xr.rio.crs, 
               'scenario': scenario, 
               'year': year, 
               'notes': 'DEM under MHHW surface denoted by -1',
               'percentile': '50'}
    )

#%%
def main():
    """
    Main function to calculate flooding days for different scenarios and years.
    """
    dem_xr = load_dem(NOAA_SLR_DEM_PATH)
    mhhw_xr = load_dem(TIDAL_SURFACE_PATH)
    mhhw_xr_aligned = mhhw_xr.rio.reproject_match(dem_xr)

    for scenario in SCENARIOS:
        scenario_dir = os.path.join(FLOOD_DAYS_RASTER_DIR, scenario)
        os.makedirs(scenario_dir, exist_ok=True)
        for year in YEARS:
            flooding_days = calculate_flooding_days(dem_xr, mhhw_xr_aligned, scenario, year)
            flooding_days.rio.to_raster(os.path.join(scenario_dir, f'{year}_{scenario}_per50.tif'))

#%%
if __name__ == '__main__':
    main()
