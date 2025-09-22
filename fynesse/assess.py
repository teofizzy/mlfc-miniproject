from typing import Any, Union
import pandas as pd
import logging

from .config import *
from . import access
import osmnx as ox
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import numpy as np
from scipy.stats import linregress
import xarray as xr


# Set up logging
logger = logging.getLogger(__name__)

"""These are the types of import we might expect in this file
import pandas
import bokeh
import seaborn
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

"""Place commands in this file to assess the data you have downloaded.
How are missing values encoded, how are outliers encoded? What do columns represent,
makes rure they are correctly labeled. How is the data indexed. Crete visualisation
routines to assess the data (e.g. in bokeh). Ensure that date formats are correct
and correctly timezoned."""


def data() -> Union[pd.DataFrame, Any]:
    """
    Load the data from access and ensure missing values are correctly encoded as well as
    indices correct, column names informative, date and times correctly formatted.
    Return a structured data structure such as a data frame.

    IMPLEMENTATION GUIDE FOR STUDENTS:
    ==================================

    1. REPLACE THIS FUNCTION WITH YOUR DATA ASSESSMENT CODE:
       - Load data using the access module
       - Check for missing values and handle them appropriately
       - Validate data types and formats
       - Clean and prepare data for analysis

    2. ADD ERROR HANDLING:
       - Handle cases where access.data() returns None
       - Check for data quality issues
       - Validate data structure and content

    3. ADD BASIC LOGGING:
       - Log data quality issues found
       - Log cleaning operations performed
       - Log final data summary

    4. EXAMPLE IMPLEMENTATION:
       df = access.data()
       if df is None:
           print("Error: No data available from access module")
           return None

       print(f"Assessing data quality for {len(df)} rows...")
       # Your data assessment code here
       return df
    """
    logger.info("Starting data assessment")

    # Load data from access module
    df = access.data()

    # Check if data was loaded successfully
    if df is None:
        logger.error("No data available from access module")
        print("Error: Could not load data from access module")
        return None

    logger.info(f"Assessing data quality for {len(df)} rows, {len(df.columns)} columns")

    try:
        # STUDENT IMPLEMENTATION: Add your data assessment code here

        # Example: Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.info(f"Found missing values: {missing_counts.to_dict()}")
            print(f"Missing values found: {missing_counts.sum()} total")

        # Example: Check data types
        logger.info(f"Data types: {df.dtypes.to_dict()}")

        # Example: Basic data cleaning (students should customize this)
        # Remove completely empty rows
        df_cleaned = df.dropna(how="all")
        if len(df_cleaned) < len(df):
            logger.info(f"Removed {len(df) - len(df_cleaned)} completely empty rows")

        logger.info(f"Data assessment completed. Final shape: {df_cleaned.shape}")
        return df_cleaned

    except Exception as e:
        logger.error(f"Error during data assessment: {e}")
        print(f"Error assessing data: {e}")
        return None


def query(data: Union[pd.DataFrame, Any]) -> str:
    """Request user input for some aspect of the data."""
    raise NotImplementedError


def view(data: Union[pd.DataFrame, Any]) -> None:
    """Provide a view of the data that allows the user to verify some aspect of its quality."""
    raise NotImplementedError


def labelled(data: Union[pd.DataFrame, Any]) -> Union[pd.DataFrame, Any]:
    """Provide a labelled set of data ready for supervised learning."""
    raise NotImplementedError

def get_osm_features(bbox, place_name, tags):
    pois = ox.features_from_bbox(bbox, tags)
    area = None
    if place_name:
        area = ox.geocode_to_gdf(place_name)
    graph = ox.graph_from_bbox(bbox)
    nodes, edges = ox.graph_to_gdfs(graph)

    return pois, area, nodes, edges

def get_pois_df(pois):
    pois_df = pd.DataFrame(pois)
    pois_df['latitude'] = pois_df.apply(lambda row: row.geometry.centroid.y, axis=1)
    pois_df['longitude'] = pois_df.apply(lambda row: row.geometry.centroid.x, axis=1)
    return pois_df

def get_feature_vector(latitude, longitude, box_size_km=2, features=None):
    """
    Given a central point (latitude, longitude) and a bounding box size,
    query OpenStreetMap via OSMnx and return a feature vector.

    Parameters
    ----------
    latitude : float
        Latitude of the center point.
    longitude : float
        Longitude of the center point.
    box_size : float
        Size of the bounding box in kilometers
    features : list of tuples
        List of (key, value) pairs to count. Example:
        [
            ("amenity", None),
            ("amenity", "school"),
            ("shop", None),
            ("tourism", "hotel"),
        ]

    Returns
    -------
    feature_vector : dict
        Dictionary of feature counts, keyed by (key, value).
    """
    from osmnx.features import InsufficientResponseError

    bbox = access.get_osm_datapoints(latitude, longitude)

    # Query OSMnx for features
    tags = {}
    for feature in features:
        tags[feature[0]] = True
    try:
      pois = get_osm_features(bbox, None, tags)[0]
    except InsufficientResponseError:
      return {}

    # Count features matching each (key, value) in poi_types
    pois_df = get_pois_df(pois)

    # Return dictionary of counts
    poi_counts = {}

    for key, value in features:
        if key in pois_df.columns:
            if value:  # count only that value
                poi_counts[f"{key}:{value}"] = (pois_df[key] == value).sum()
            else:  # count any non-null entry
                poi_counts[key] = pois_df[key].notnull().sum()
        else:
            poi_counts[f"{key}:{value}" if value else key] = 0

    return poi_counts

def build_feature_dataframe(city_dicts, features, box_size_km=1):
    results = {}
    for country, cities in city_dicts:
        for city, coords in cities.items():
            vec = get_feature_vector(
                coords["latitude"],
                coords["longitude"],
                box_size_km=box_size_km,
                features=features
            )
            vec["country"] = country
            results[city] = vec
    return pd.DataFrame(results).T

def visualize_feature_space(df, X, label1, label2, label1_color, label2_color):
    pca = PCA(n_components=2)
    X_proj = pca.fit_transform(X)
    plt.figure(figsize=(8,6))
    for label, color in [(label1, label1_color), (label2, label2_color)]:
        mask = (y == label)
        plt.scatter(X_proj[mask, 0], X_proj[mask, 1],
                    label=label, color=color, s=100, alpha=0.7)

    for i, feature in enumerate(df.index):
        plt.text(X_proj[i,0]+0.02, X_proj[i,1], feature, fontsize=8)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("2D projection of feature vectors")
    plt.legend()
    plt.show()

def plot_city_map(city, country, latitude=None, longitude=None, box_size=2, plot_from_place=False):
    """
    Plot a map of a city with points of interest.
    """
    tags = {
    "amenity": True,
    "buildings": True,
    "historic": True,
    "leisure": True,
    "shop": True,
    "tourism": True,
    "religion": True,
    "memorial": True
    }
    place_name = f"{city}, {country}"

    if latitude or longitude:
        # Get bbox
        bbox = access.get_osm_datapoints(latitude, longitude)
        pois = ox.features_from_bbox(bbox, tags=tags)

        # Get graph elements
        graph = ox.graph_from_bbox(bbox)
        area = ox.geocode_to_gdf(place_name)
        nodes, edges = ox.graph_to_gdfs(graph)
        buildings = ox.features_from_bbox(bbox, tags={"building": True})

    if plot_from_place:
        graph = ox.graph_from_place(place_name, network_type='all')
        pois = ox.features_from_place(place_name, tags=tags)
        area = ox.geocode_to_gdf(place_name)
        nodes, edges = ox.graph_to_gdfs(graph)
        buildings = ox.features_from_place(place_name, tags={"building": True})

    # Plot the city map
    fig, ax = plt.subplots(figsize=(6,6))
    area.plot(ax=ax, color="tan", alpha=0.5)
    buildings.plot(ax=ax, facecolor="gray", edgecolor="gray")
    edges.plot(ax=ax, linewidth=1, edgecolor="black", alpha=0.3)
    nodes.plot(ax=ax, color="black", markersize=1, alpha=0.3)
    pois.plot(ax=ax, color="green", markersize=5, alpha=1)
    ax.set_xlim(bbox[0], bbox[2])
    ax.set_ylim(bbox[1], bbox[3])
    ax.set_title(place_name, fontsize=14)
    plt.show()

def plot_trend_map(trend, title, cmap="coolwarm"):
    """Quick map of linear trend over Africa."""
    plt.figure(figsize=(8,6))
    trend.plot(cmap=cmap)
    plt.title(title)
    plt.show()

def plot_region_timeseries(da, title, label):
    """Plot regional mean timeseries."""
    weighted_da = get_latitude_weights(da)
    ts = weighted_da.mean(["latitude", "longitude"])
    ts.plot.line(label=label)
    plt.title(f"{label} region timeseries")
    plt.xlabel("Time")
    plt.ylabel(f"{da.attrs['long_name']} ({da.attrs['units']})")
    plt.legend()
    plt.show()

def plot_decadal_boxplot(da, varname):
    """Boxplot of decadal variability for a region."""
    weighted_da = get_latitude_weights(da)
    ts = weighted_da.mean(["latitude", "longitude"])
    df = ts.to_dataframe().reset_index()
    df["decade"] = (df["time"].dt.year // 10) * 10
    df.boxplot(column=varname, by="decade", figsize=(8,6))
    plt.title(f"{varname} by decade")
    plt.ylabel(f"{da.attrs['long_name']} ({da.attrs['units']})")
    plt.suptitle("")
    plt.show()

def plot_climatology(mean, std, region):
    """
    Plot monthly climatology
    """
    fig, ax = plt.subplots(1, 1, figsize = (12, 6))

    ax.plot(mean.month, mean, color='blue', label='mean')
    ax.fill_between(mean.month, (mean + std), (mean - std), alpha=0.1, color='green', label='+/- 1 SD')
    ax.set_title(f'{region} monthly climatology of 2m temperature')
    ax.set_ylabel('° C')
    ax.set_xlabel('month')
    ax.set_xlim(1,12)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)

def get_latitude_weights(da):
    """
    Compute the cosine of latitude weights.
    """
    weights = np.cos(np.deg2rad(da.latitude))
    weights.name = "weights"
    da_weighted = da.weighted(weights)
    return da_weighted

def compute_anomalies(da, baseline=(1991, 2020)):
    """Compute anomalies relative to baseline climatology."""
    baseline_ds = da.sel(time=slice(f"{baseline[0]}-01-01", f"{baseline[1]}-12-31"))
    clim = baseline_ds.groupby("time.month").mean("time")
    anomalies = da.groupby("time.month") - clim
    return anomalies

def monthly_climatology(da, start_date:str, end_date:str,):
    """
    Compute monthly climatology.
    """
    clim_period = da.sel(time=slice(start_date, end_date))
    clim_mean = clim_period.groupby('time.month').mean()
    clim_std = clim_period.groupby('time.month').std()

    clim_mean_weighted = get_latitude_weights(clim_mean)
    clim_std_weighted = get_latitude_weights(clim_std)
    mean = clim_mean_weighted.mean(dim=['latitude', 'longitude'])
    std = clim_std_weighted.mean(dim=['latitude', 'longitude'])

    return mean, std

def compute_linear_trend(da, dim="time"):
    """
    Compute linear trend (slope per year) at each grid cell.
    """
    # Convert time to numeric (e.g. years since start)
    time_num = xr.DataArray(
        np.arange(len(da[dim])),
        dims=dim,
        coords={dim: da[dim]}
    )

    def _polyfit(y, x):
        mask = np.isfinite(y)
        if mask.sum() < 2:
            return np.nan
        slope, intercept = np.polyfit(x[mask], y[mask], 1)
        return slope

    slope = xr.apply_ufunc(
        _polyfit, da, time_num,
        input_core_dims=[[dim], [dim]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float]
    )

    return slope

def get_int(year, month, first_year=1979):
    """
    Given a year and month, return the corresponding integer value.
    """
    if year < first_year:
        raise ValueError(f"Year must be greater than or equal to {first_year}")
    if month < 1 or month > 12:
        raise ValueError("Month must be between 1 and 12")

    return (year - first_year) * 12 + month - 1

def plot_da(da, year:int, month:int, label:str):
    da[get_int(year, month), :, :].plot()
    plt.suptitle(f"{da.attrs['long_name']} for {label}")
    plt.show()

def compute_regional_mean(da):
    # Average over space
    weighted_da = get_latitude_weights(da)

    # Make regional da 1D: Averaging over space
    region_mean = weighted_da.mean(dim=["latitude", "longitude"])

    return region_mean

def plot_region_timeseries(da, label):
    # Fit linear regression
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import linregress

    da.plot(label=label)
    time_num = np.arange(len(da["time"]))
    slope, intercept, r, p, se = linregress(time_num, da.values)

    plt.plot(da["time"], intercept + slope*time_num, "r--", label="Linear Trend")
    plt.title(f"{label} region mean timeseries")
    plt.xlabel("Time")
    # plt.ylabel(f"{da.attrs['long_name']} ({da.attrs['units']})")
    plt.legend()
    plt.show()

def summarize_region(da, name, baseline=(1991, 2020)):
    reg_mean = compute_regional_mean(da)  # 1D time series

    # Linear trend (per decade)
    time_num = np.arange(len(reg_mean["time"])) / 12  # in years
    slope, intercept, r, p, se = linregress(time_num, reg_mean.values)
    slope_decade = slope * 10  # °C per decade

    # Mean and variability
    mean_val = float(reg_mean.mean().values)
    std_val = float(reg_mean.std().values)

    # Anomalies relative to baseline
    anomalies = compute_anomalies(da, baseline=baseline)
    anomalies_mean = compute_regional_mean(anomalies)
    recent_anom = anomalies_mean.sel(time=slice("2014-01-01", "2023-12-31")).mean().values

    return {
        "Region": name,
        "Trend (°C/decade)": slope_decade,
        "Mean (°C)": mean_val,
        "StdDev (°C)": std_val,
        "Recent anomaly vs 1991–2020 (°C)": recent_anom
    }
