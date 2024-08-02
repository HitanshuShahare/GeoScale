from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import interpolate
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
import io
from flask_cors import CORS
from PIL import Image
import base64
import logging
import fiona
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import MinMaxScaler
from pykalman import KalmanFilter
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.linalg import hankel
import geopandas as gpd
from shapely.geometry import Point
import os

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set the SHAPE_RESTORE_SHX environment variable
os.environ['SHAPE_RESTORE_SHX'] = 'YES'

# Define the spatial interpolation functions
def inverse_distance_weighting(x, y, z, xi, yi):
    grid_z = interpolate.griddata((x, y), z, (xi, yi), method='cubic', fill_value=np.nan)
    return grid_z

def ordinary_kriging(x, y, z, xi, yi):
    OK = OrdinaryKriging(x, y, z, variogram_model='spherical')
    z_interpolated, _ = OK.execute('points', xi, yi)
    return z_interpolated

def radial_basis_function(x, y, z, xi, yi):
    rbf = interpolate.Rbf(x, y, z, function='multiquadric')
    z_interpolated = rbf(xi, yi)
    return z_interpolated

def nearest_neighbor_interpolation(x, y, z, xi, yi):
    grid_z = interpolate.griddata((x, y), z, (xi, yi), method='nearest')
    return grid_z

def linear_interpolation_spatial(x, y, z, xi, yi):
    grid_z = interpolate.griddata((x, y), z, (xi, yi), method='linear', fill_value=np.nan)
    return grid_z

def natural_neighbor_interpolation(x, y, z, xi, yi):
    return linear_interpolation_spatial(x, y, z, xi, yi)  # Fallback to linear interpolation

def thin_plate_spline(x, y, z, xi, yi):
    tps = interpolate.Rbf(x, y, z, function='thin_plate')
    z_interpolated = tps(xi, yi)
    return z_interpolated

def universal_kriging(x, y, z, xi, yi):
    UK = UniversalKriging(x, y, z, variogram_model='linear')
    z_interpolated, _ = UK.execute('points', xi, yi)
    return z_interpolated

def cokriging(x, y, z, secondary_variable, xi, yi):
    return ordinary_kriging(x, y, z, xi, yi)  # Fallback to ordinary kriging

def trend_surface_analysis(x, y, z, xi, yi):
    A = np.c_[np.ones(x.shape), x, y, x*y, x**2, y**2]
    C, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
    zi = C[0] + C[1]*xi + C[2]*yi + C[3]*xi*yi + C[4]*xi**2 + C[5]*yi**2
    return zi

def generate_temporal_map(df, technique, time_step):
    return Image.new('RGB', (100, 100)), 0

@app.route('/generate_climate_map', methods=['POST'])
def generate_climate_map():
    try:
        file = request.files['file']
        interpolation_type = request.form['interpolationType']
        technique = request.form['technique']
        time_step = int(request.form['timeStep'])

        df = pd.read_csv(file)

        if interpolation_type == 'spatial':
            image, max_time_step = generate_spatial_map(df, technique)
        else:  # temporal
            image, max_time_step = generate_temporal_map(df, technique, time_step)

        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return jsonify({
            'image': f"data:image/png;base64,{img_str}",
            'maxTimeStep': max_time_step
        })
    except Exception as e:
        logger.error(f"Error in generate_climate_map: {str(e)}")
        return jsonify({'error': str(e)}), 400

def generate_spatial_map(df, technique):
    times = df['time'].unique()
    max_time_step = len(times) - 1

    df_time = df[df['time'] == times[-1]]
    x = df_time['longitude'].values
    y = df_time['latitude'].values
    z = df_time['temperature'].values

    # Load the shapefile or GeoJSON for the regions
    shapefile_path = r'C:\Users\91876\Downloads\shapefile_india\india.shp'
    regions = gpd.read_file(shapefile_path)

    # Check if regions has a CRS, if not set it to WGS84
    if regions.crs is None:
        regions = regions.set_crs("EPSG:4326")

    # Create points GeoDataFrame with explicit CRS
    points = gpd.GeoDataFrame(
        df_time, 
        geometry=gpd.points_from_xy(df_time.longitude, df_time.latitude),
        crs="EPSG:4326"  # WGS84 coordinate system
    )

    # Ensure the regions and points have the same CRS
    regions = regions.to_crs(points.crs)

    # Perform spatial interpolation
    grid_x, grid_y = np.meshgrid(np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100))
    xi, yi = grid_x.flatten(), grid_y.flatten()

    interpolation_functions = {
        'idw': inverse_distance_weighting,
        'kriging': ordinary_kriging,
        'rbf': radial_basis_function,
        'nearest': nearest_neighbor_interpolation,
        'linear': linear_interpolation_spatial,
        'natural_neighbor': natural_neighbor_interpolation,
        'thin_plate_spline': thin_plate_spline,
        'universal_kriging': universal_kriging,
        'cokriging': cokriging,
        'trend_surface': trend_surface_analysis
    }

    if technique not in interpolation_functions:
        raise ValueError(f"Unknown spatial interpolation technique: {technique}")

    z_interpolated = interpolation_functions[technique](x, y, z, xi, yi)
    z_interpolated = z_interpolated.reshape(grid_x.shape)

    # Create a DataFrame for the interpolated grid
    interpolated_df = pd.DataFrame({'longitude': xi, 'latitude': yi, 'temperature': z_interpolated.flatten()})
    interpolated_points = gpd.GeoDataFrame(
        interpolated_df, 
        geometry=gpd.points_from_xy(interpolated_df.longitude, interpolated_df.latitude),
        crs=points.crs
    )

    # Spatial join between the interpolated points and the regions
    joined = gpd.sjoin(interpolated_points, regions, how="left", predicate="within")

    # Aggregate by region
    region_means = joined.groupby('region_id')['temperature'].mean().reset_index()

    # Merge back to the regions GeoDataFrame
    regions = regions.merge(region_means, on='region_id', how='left')

    # Plot the choropleth map
    plt.figure(figsize=(10, 6))
    regions.plot(column='temperature', cmap='viridis', legend=True)
    plt.title(f'Climate Map using {technique.capitalize()}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close()

    return Image.open(buf), max_time_step

if __name__ == '__main__':
    app.run(debug=True)