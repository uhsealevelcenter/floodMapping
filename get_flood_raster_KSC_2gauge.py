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
import os
import pandas as pd
import rasterio as rio
import rioxarray

# Constants
NOAA_SLR_DEM_PATH = './inputData/NOAA_SLR_DEM/NOAA_SLR_DEM_J995345.tif'
TIDAL_SURFACE_PATH = './tidal_surface/tidal_surface.tif'
TRIDENT_MHHW = 0.336  # meters MHHW from NOAA datum info (https://tidesandcurrents.noaa.gov/datums.html?datum=NAVD88&units=1&epoch=0&id=8721604&name=Trident+Pier%2C+Port+Canaveral&state=FL)
SCENARIOS = ['high', 'int', 'int_high', 'int_low', 'low']
YEARS = np.arange(2020, 2101, 10)  # Decadal intervals
FLOOD_DAYS_RASTER_DIR = './flood_days_raster'
TGDATA_DIR_COAST = './inputData/8721604_MHHW'
TGDATA_DIR_INLAND = './inputData/02248380_MHHW'

#%%
SCENARIOS = ['high', 'int']
YEARS = np.arange(2020, 2030, 10)  # Decadal intervals
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
# Calculate which pixels are inland and which are coastal


#%%
def threshold_to_days(scenario,TG_SOURCE):
    """
    Calculate the number of flooding days at different thresholds for a given scenario.

    Parameters:
    - scenario (str): The name of the scenario.

    Returns:
    - df (pandas.DataFrame): A DataFrame containing the number of flooding days at different thresholds.
    """

    totals_by_threshold = {}
    for threshold in np.arange(-0.34, 3.30, 0.01):  # Thresholds from 0.2 to 3 meters
        try:
            with open(os.path.join(TG_SOURCE, f'{scenario}/{int(threshold * 100):03d}.json')) as f:
                data = json.load(f)
            totals_by_threshold[np.round(threshold,2)] = data['annual_percentiles']['percentiles']['50']
        except:
            totals_by_threshold[np.round(threshold,2)] = 365


    # turn this into a dataframe
    df = pd.DataFrame(totals_by_threshold)
    df.index = np.arange(2020, 2101, 1)
    return df

#%%
def calculate_flooding_days(dem_xr, mhhw_xr_aligned, scenario, year, TG_SOURCE):
    """
    Calculate the number of flooding days for a given Digital Elevation Model (DEM) dataset,
    scenario, and year.

    Parameters:
        dem_xr (xarray.DataArray): The Digital Elevation Model dataset in NAVD88.
        mhhw_xr_aligned (xarray.DataArray): The Mean Higher High Water (MHHW) surface aligned with the DEM.
        scenario (str): The flood scenario.
        year (int): The year for which to calculate the flooding days.
        TG_SOURCE (str): The path to the threshold gauge data.

    Returns:
        xarray.DataArray: A DataArray containing the number of flooding days for each elevation
        in the DEM dataset, with special values assigned for land under MHHW, ocean, and no data.

    """
    df = threshold_to_days(scenario,TG_SOURCE)

    # subtract the MHHW from the DEM  - we are now working in MHHW space 
    elevations = dem_xr.values - mhhw_xr_aligned.values
    elevations = np.round(elevations, 2)



    
    days_per_year = np.full_like(elevations, np.nan)
    for elevation, days in df.loc[year].items():
        days_per_year[elevations == elevation] = days        

    days_per_year[elevations>elevation] = 0

    #set nans in days_per_year to 0
    days_per_year[np.isnan(days_per_year)] = -50    

    # Assign special values for land under MHHW and ocean
    # days_per_year[elevations<=0] = -1  # Land under 0 NAVD88
    days_per_year[dem_xr.values<=-99] = -99  # No data

    

    return xr.DataArray(
        days_per_year,
        coords=dem_xr.coords,
        dims=dem_xr.dims,
        attrs={'crs': dem_xr.rio.crs, 
               'scenario': scenario, 
               'year': year, 
            #    'notes': 'DEM under MHHW surface denoted by -1',
               'percentile': '50',
               'flood_source': TG_SOURCE}
    ), elevations


#%%
def calculate_flooding_days_with_mask(dem_xr, mhhw_xr_aligned, scenario, year):
    """
    Calculate the number of flooding days for a given Digital Elevation Model (DEM) dataset,
    scenario, and year.

    Parameters:
        dem_xr (xarray.DataArray): The Digital Elevation Model dataset in NAVD88.
        mhhw_xr_aligned (xarray.DataArray): The Mean Higher High Water (MHHW) surface aligned with the DEM.
        scenario (str): The flood scenario.
        year (int): The year for which to calculate the flooding days.
        TG_SOURCE (str): The paths to the threshold gauge data.

    Returns:
        xarray.DataArray: A DataArray containing the number of flooding days for each elevation
        in the DEM dataset, with special values assigned for land under MHHW, ocean, and no data.

    """
    df_Coast = threshold_to_days(scenario,TGDATA_DIR_COAST)
    df_Inland = threshold_to_days(scenario,TGDATA_DIR_INLAND)

    # get the elevation where the threshold is 365 days
    df_Inland_flood = df_Inland.loc[year][df_Inland.loc[year] == 365].last_valid_index()
    df_Coast_flood = df_Coast.loc[year][df_Coast.loc[year] == 365].last_valid_index()
    # disconnected =1, inland = 2, coast = 3

    elevations = dem_xr.values - mhhw_xr_aligned.values
    elevations = np.round(elevations, 2)

    days_per_year = np.full_like(elevations, np.nan)

    for elevation in np.arange(-0.34, 3.3, 0.01):
        elevation = np.round(elevation, 2)
        mask_file = f'./connected_masks/mask_combined_{elevation:.2f}mMHHW.nc'
        with xr.open_dataset(mask_file) as mask_ds:
            mask_combined = mask_ds['mask_combined'].values
            # mask_combined[(mask_combined == 0) & (elevations <= elevation)] = 1
            days_per_year[(mask_combined == 1) & (elevations == elevation)] = df_Inland.loc[year][elevation] #assume Inland-connected for disconnected
                #here is where we put the ZONE OF INFLUENCE!!
            days_per_year[(mask_combined == 2) & (elevations == elevation)] = df_Inland.loc[year][elevation]
            days_per_year[(mask_combined == 3) & (elevations == elevation)] = df_Coast.loc[year][elevation]
            # days_per_year[(mask_combined == 1) & np.isnan(days_per_year) & (elevations+0.01 == elevation)] = df_Inland.loc[year][elevation]
            days_per_year[(elevations <= df_Inland_flood) & np.isnan(days_per_year) & (mask_combined == 0)] = 500
            days_per_year[(elevations <= df_Inland_flood) & np.isnan(days_per_year) & (mask_combined == 2)] = 1000

    days_per_year = xr.DataArray(
                         days_per_year,
                         coords=dem_xr.coords,
                         dims=dem_xr.dims,
                         attrs={'crs': dem_xr.rio.crs, 
                                'scenario': scenario, 
                                'year': year, 
                                'percentile': '50'}
                     )

    days_per_year = days_per_year.where(elevations < 3.01, 0) #set elevations above 3.01m to 0 days per year
    # days_per_year = days_per_year.where((elevations >= df_Inland_flood) & ~np.isnan(days_per_year), 1000) #set elevations below RSL level to 1000
    days_per_year = days_per_year.where(dem_xr != -99, -99)
    days_per_year = days_per_year.where(dem_xr != -999999, -999999)



    return days_per_year
                   
#%%
def main():
    """
    Main function to calculate flooding days for different scenarios and years.
    """
    dem_xr = load_dem(NOAA_SLR_DEM_PATH)
    mhhw_xr = load_dem(TIDAL_SURFACE_PATH)
    mhhw_xr_aligned = mhhw_xr.rio.reproject_match(dem_xr)

    dem_xr = dem_xr.isel(y=slice(300, dem_xr.shape[0]), x=slice(0,dem_xr.shape[1]-300))
    mhhw_xr_aligned = mhhw_xr_aligned.isel(y=slice(300, mhhw_xr_aligned.shape[0]), x=slice(0,mhhw_xr_aligned.shape[1]-300))


    for scenario in SCENARIOS:
        scenario_dir = os.path.join(FLOOD_DAYS_RASTER_DIR, scenario)
        os.makedirs(scenario_dir, exist_ok=True)
        for year in YEARS:
            print(year, scenario)
            flooding_days = calculate_flooding_days_with_mask(dem_xr, mhhw_xr_aligned, scenario, year)
            flooding_days.rio.to_raster(os.path.join(scenario_dir, f'{year}_{scenario}_per50.tif'))

             
#%%
if __name__ == '__main__':
    main()
#%%

