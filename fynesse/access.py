"""
Access module for the fynesse framework.

This module handles data access functionality including:
- Data loading from various sources (web, local files, databases)
- Legal compliance (intellectual property, privacy rights)
- Ethical considerations for data usage
- Error handling for access issues

Legal and ethical considerations are paramount in data access.
Ensure compliance with e.g. .GDPR, intellectual property laws, and ethical guidelines.

Best Practice on Implementation
===============================

1. BASIC ERROR HANDLING:
   - Use try/except blocks to catch common errors
   - Provide helpful error messages for debugging
   - Log important events for troubleshooting

2. WHERE TO ADD ERROR HANDLING:
   - File not found errors when loading data
   - Network errors when downloading from web
   - Permission errors when accessing files
   - Data format errors when parsing files

3. SIMPLE LOGGING:
   - Use print() statements for basic logging
   - Log when operations start and complete
   - Log errors with context information
   - Log data summary information

4. EXAMPLE PATTERNS:
   
   Basic error handling:
   try:
       df = pd.read_csv('data.csv')
   except FileNotFoundError:
       print("Error: Could not find data.csv file")
       return None
   
   With logging:
   print("Loading data from data.csv...")
   try:
       df = pd.read_csv('data.csv')
       print(f"Successfully loaded {len(df)} rows of data")
       return df
   except FileNotFoundError:
       print("Error: Could not find data.csv file")
       return None
"""

from typing import Any, Union
import pandas as pd
import logging
import osmnx as ox
import matplotlib.pyplot as plt
import os
import numpy as np
import xarray as xr
import cdsapi


# Set up basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def data(file_path:str) -> Union[pd.DataFrame, None]:
    """
    Read the data from the web or local file, returning structured format such as a data frame.

    IMPLEMENTATION GUIDE
    ====================

    1. REPLACE THIS FUNCTION WITH YOUR ACTUAL DATA LOADING CODE:
       - Load data from your specific sources
       - Handle common errors (file not found, network issues)
       - Validate that data loaded correctly
       - Return the data in a useful format

    2. ADD ERROR HANDLING:
       - Use try/except blocks for file operations
       - Check if data is empty or corrupted
       - Provide helpful error messages

    3. ADD BASIC LOGGING:
       - Log when you start loading data
       - Log success with data summary
       - Log errors with context

    4. EXAMPLE IMPLEMENTATION:
       try:
           print("Loading data from data.csv...")
           df = pd.read_csv('data.csv')
           print(f"Successfully loaded {len(df)} rows, {len(df.columns)} columns")
           return df
       except FileNotFoundError:
           print("Error: data.csv file not found")
           return None
       except Exception as e:
           print(f"Error loading data: {e}")
           return None

    Returns:
        DataFrame or other structured data format
    """
    logger.info("Starting data access operation")

    try:
        # IMPLEMENTATION: Replace this with your actual data loading code
        # Example: Load data from a CSV file
        logger.info("Loading data from data.csv")
        df = pd.read_csv(file_path)

        # Basic validation
        if df.empty:
            logger.warning("Loaded data is empty")
            return None

        logger.info(
            f"Successfully loaded data: {len(df)} rows, {len(df.columns)} columns"
        )
        return df

    except FileNotFoundError:
        logger.error("Data file not found: data.csv")
        print("Error: Could not find data.csv file. Please check the file path.")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}")
        print(f"Error loading data: {e}")
        return None


def get_osm_datapoints(latitude, longitude, use_km=True, box_size_km=2):
    
    if use_km:
        # Define box width and height
        box_width = box_size_km/111 # 111km = 1 degree at the equator
        box_height = box_size_km/111
    else:
        box_height = 0.02
        box_width = 0.02

    north = latitude + box_height/2
    south = latitude - box_height/2
    west = longitude - box_width/2
    east = longitude + box_width/2

    # Get bounding box
    bbox = (west, south, east, north)
    return bbox

def download_monthly_data(client, year:list[str],time:list[str], month:list[str],
                          area:list[float], dataset:str, variable:str,
                          product_type:str, destination_dir:str, filename:str):
    """
    Download monthly data from the CDS API.
    """
    # ensure destination_directory
    os.makedirs(destination_dir, exist_ok=True)
    file_path = os.path.join(destination_dir, filename)

    # Check if the file already exists
    if not os.path.exists(file_path):
        try:
            # Retrieve data
            client.retrieve(
                dataset,
                {
                    'product_type': f'{product_type}',
                    'variable': f'{variable}',
                    'year': year,
                    'month': month,
                    'time': time,
                    'area': area,
                    'data_format': 'netcdf',
                },
            os.path.join(destination_dir, filename)
            )
        except Exception as e:
            print(f"Error downloading data: {e}")

    else:
        print(f"File {file_path} already exists. Skipping download.")

def subset_region(ds, region_bbox:list, latitude_var_name, longitude_var_name):
    """
    Subset a dataset to a specific region defined by a bounding box.
    The distance between box height and box width should be at least 0.25 degrees
    due to the spatial resolution of the dataset

    region_bbox: max_lat, min_lon, min_lat, max_lon [north, west, south, east]
    """
    ds_sliced = ds.sel(
        **{latitude_var_name: slice(region_bbox[0], region_bbox[2]),
           longitude_var_name: slice(region_bbox[1], region_bbox[3])}
    )

    return ds_sliced

def download_monthly_data(client, year:list[str],time:list[str], month:list[str],
                          area:list[float], dataset:str, variable:str,
                          product_type:str, destination_dir:str, filename:str):
    """
    Download monthly data from the CDS API.
    """
    # ensure destination_directory
    os.makedirs(destination_dir, exist_ok=True)
    file_path = os.path.join(destination_dir, filename)

    # Check if the file already exists
    if not os.path.exists(file_path):
        try:
            # Retrieve data
            client.retrieve(
                dataset,
                {
                    'product_type': f'{product_type}',
                    'variable': f'{variable}',
                    'year': year,
                    'month': month,
                    'time': time,
                    'area': area,
                    'data_format': 'netcdf',
                },
            os.path.join(destination_dir, filename)
            )
        except Exception as e:
            print(f"Error downloading data: {e}")

    else:
        print(f"File {file_path} already exists. Skipping download.")

def subset_region(ds, region_bbox:list, latitude_var_name, longitude_var_name):
    """
    Subset a dataset to a specific region defined by a bounding box.
    The distance between box height and box width should be at least 0.25 degrees
    due to the spatial resolution of the dataset

    region_bbox: max_lat, min_lon, min_lat, max_lon [north, west, south, east]
    """
    ds_sliced = ds.sel(
        **{latitude_var_name: slice(region_bbox[0], region_bbox[2]),
           longitude_var_name: slice(region_bbox[1], region_bbox[3])}
    )

    return ds_sliced
