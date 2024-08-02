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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import MinMaxScaler
from pykalman import KalmanFilter
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.linalg import hankel

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Spatial interpolation methods

def inverse_distance_weighting(x, y, z, xi, yi):
    dist = np.sqrt((x[:, np.newaxis] - xi)**2 + (y[:, np.newaxis] - yi)**2)
    weights = 1.0 / (dist**2 + 1e-8)
    return np.sum(weights * z[:, np.newaxis], axis=0) / np.sum(weights, axis=0)

def ordinary_kriging(x, y, z, xi, yi):
    try:
        OK = OrdinaryKriging(x, y, z, variogram_model='spherical', verbose=False, enable_plotting=False)
        z_interpolated, _ = OK.execute('points', xi, yi)
        return z_interpolated
    except Exception as e:
        logger.error(f"Error in ordinary kriging: {str(e)}")
        return np.full_like(xi, np.nan)

def radial_basis_function(x, y, z, xi, yi):
    try:
        rbf = interpolate.Rbf(x, y, z, function='multiquadric')
        return rbf(xi, yi)
    except Exception as e:
        logger.error(f"Error in radial basis function: {str(e)}")
        return np.full_like(xi, np.nan)

def nearest_neighbor_interpolation(x, y, z, xi, yi):
    try:
        return interpolate.griddata((x, y), z, (xi, yi), method='nearest')
    except Exception as e:
        logger.error(f"Error in nearest neighbor interpolation: {str(e)}")
        return np.full_like(xi, np.nan)

def linear_interpolation_spatial(x, y, z, xi, yi):
    try:
        return interpolate.griddata((x, y), z, (xi, yi), method='linear')
    except Exception as e:
        logger.error(f"Error in linear interpolation: {str(e)}")
        return np.full_like(xi, np.nan)

def natural_neighbor_interpolation(x, y, z, xi, yi):
    try:
        return interpolate.griddata((x, y), z, (xi, yi), method='cubic')
    except Exception as e:
        logger.error(f"Error in natural neighbor interpolation: {str(e)}")
        return np.full_like(xi, np.nan)

def thin_plate_spline(x, y, z, xi, yi):
    try:
        tps = interpolate.Rbf(x, y, z, function='thin_plate')
        return tps(xi, yi)
    except Exception as e:
        logger.error(f"Error in thin plate spline: {str(e)}")
        return np.full_like(xi, np.nan)

def universal_kriging(x, y, z, xi, yi):
    try:
        UK = UniversalKriging(x, y, z, drift_terms=['regional_linear'])
        z_interpolated, _ = UK.execute('points', xi, yi)
        return z_interpolated
    except Exception as e:
        logger.error(f"Error in universal kriging: {str(e)}")
        return np.full_like(xi, np.nan)

def cokriging(x, y, z, secondary_variable, xi, yi):
    return ordinary_kriging(x, y, z, xi, yi)

def trend_surface_analysis(x, y, z, xi, yi, order=3):
    try:
        A = np.column_stack([np.ones(len(x))] + 
                            [x**i for i in range(1, order+1)] + 
                            [y**i for i in range(1, order+1)])
        coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
        
        Xi, Yi = np.meshgrid(xi, yi)
        A_interp = np.column_stack([np.ones(len(Xi.ravel()))] + 
                                   [Xi.ravel()**i for i in range(1, order+1)] + 
                                   [Yi.ravel()**i for i in range(1, order+1)])
        
        return np.dot(A_interp, coeffs).reshape(Xi.shape)
    except Exception as e:
        logger.error(f"Error in trend surface analysis: {str(e)}")
        return np.full_like(xi, np.nan)

# Temporal interpolation methods

def linear_interpolation(times, values, new_times):
    return np.interp(new_times, times, values)

def spline_interpolation(times, values, new_times):
    if len(times) > 3:
        cs = interpolate.CubicSpline(times, values)
        return cs(new_times)
    else:
        return linear_interpolation(times, values, new_times)

def arima_forecast(times, values, new_times):
    if len(times) > 30:
        model = ARIMA(values, order=(1,1,1))
        results = model.fit()
        forecast = results.forecast(steps=len(new_times))
        return forecast
    else:
        return linear_interpolation(times, values, new_times)

def exponential_smoothing(times, values, new_times):
    if len(times) > 12:
        model = ExponentialSmoothing(values, seasonal_periods=12, trend='add', seasonal='add')
        results = model.fit()
        forecast = results.forecast(len(new_times))
        return forecast
    else:
        return linear_interpolation(times, values, new_times)

def prophet_forecast(times, values, new_times):
    if len(times) > 30:
        df = pd.DataFrame({'ds': times, 'y': values})
        model = Prophet()
        model.fit(df)
        future = pd.DataFrame({'ds': new_times})
        forecast = model.predict(future)
        return forecast['yhat'].values
    else:
        return linear_interpolation(times, values, new_times)

def kalman_filter(times, values, new_times):
    if len(times) > 10:
        kf = KalmanFilter(initial_state_mean=values[0], n_dim_obs=1)
        smoothed_state_means, _ = kf.smooth(values)
        return np.interp(new_times, times, smoothed_state_means.flatten())
    else:
        return linear_interpolation(times, values, new_times)

def simple_exponential_smoothing(times, values, new_times, alpha=0.3):
    if len(times) > 10:
        result = [values[0]]
        for n in range(1, len(values)):
            result.append(alpha * values[n] + (1 - alpha) * result[n-1])
        return np.interp(new_times, times, result)
    else:
        return linear_interpolation(times, values, new_times)

def advanced_forecast(times, values, new_times):
    return simple_exponential_smoothing(times, values, new_times)

def gaussian_process_regression(times, values, new_times):
    if len(times) > 10:
        kernel = RBF(length_scale=1.0)
        gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)
        gpr.fit(times.reshape(-1, 1), values)
        return gpr.predict(new_times.reshape(-1, 1))
    else:
        return linear_interpolation(times, values, new_times)

def singular_spectrum_analysis(times, values, new_times, window=10):
    if len(times) > window * 2:
        # Embed the time series
        X = hankel(values[:(len(values) - window + 1)], values[(len(values) - window):])
        
        # Perform SVD
        U, s, V = np.linalg.svd(X)
        
        # Reconstruct using first 3 components
        X_rec = (U[:, :3] @ np.diag(s[:3]) @ V[:3, :])
        
        # Extract the reconstructed series
        reconstructed = np.mean(hankel(X_rec[:, 0], X_rec[-1, :]), axis=1)
        
        return np.interp(new_times, times, reconstructed)
    else:
        return linear_interpolation(times, values, new_times)

def sarima_forecast(times, values, new_times):
    if len(times) > 50:
        model = SARIMAX(values, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        results = model.fit()
        forecast = results.forecast(steps=len(new_times))
        return forecast
    else:
        return linear_interpolation(times, values, new_times)

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

    grid_x, grid_y = np.meshgrid(np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100))
    xi, yi = grid_x.flatten(), grid_y.flatten()

    try:
        if technique == 'idw':
            z_interpolated = inverse_distance_weighting(x, y, z, xi, yi)
        elif technique == 'kriging':
            z_interpolated = ordinary_kriging(x, y, z, xi, yi)
        elif technique == 'rbf':
            z_interpolated = radial_basis_function(x, y, z, xi, yi)
        elif technique == 'nearest':
            z_interpolated = nearest_neighbor_interpolation(x, y, z, xi, yi)
        elif technique == 'linear':
            z_interpolated = linear_interpolation_spatial(x, y, z, xi, yi)
        elif technique == 'natural_neighbor':
            z_interpolated = natural_neighbor_interpolation(x, y, z, xi, yi)
        elif technique == 'thin_plate_spline':
            z_interpolated = thin_plate_spline(x, y, z, xi, yi)
        elif technique == 'universal_kriging':
            z_interpolated = universal_kriging(x, y, z, xi, yi)
        elif technique == 'cokriging':
            secondary_variable = df_time['secondary_variable'].values if 'secondary_variable' in df_time.columns else None
            z_interpolated = cokriging(x, y, z, secondary_variable, xi, yi)
        elif technique == 'trend_surface':
            z_interpolated = trend_surface_analysis(x, y, z, xi, yi)
        else:
            raise ValueError(f"Unknown spatial interpolation technique: {technique}")

        z_interpolated = z_interpolated.reshape(grid_x.shape)
    except Exception as e:
        logger.error(f"Error in spatial interpolation: {str(e)}")
        z_interpolated = np.full_like(grid_x, np.nan)

    plt.figure(figsize=(10, 6))
    plt.contourf(grid_x, grid_y, z_interpolated, cmap='viridis', levels=20)
    plt.colorbar(label='Temperature (°C)')
    plt.title(f'Climate Map using {technique.capitalize()}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close()

    return Image.open(buf), max_time_step

def generate_temporal_map(df, technique, time_step):
    times = df['time'].unique()
    max_time_step = len(times) - 1

    df_time = df[df['time'] == times[time_step]]
    x = df_time['longitude'].values
    y = df_time['latitude'].values

    grid_x, grid_y = np.meshgrid(np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100))
    z_interpolated = np.zeros_like(grid_x)

    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            idx = np.argmin((df['longitude'] - grid_x[i, j])**2 + (df['latitude'] - grid_y[i, j])**2)
            time_series = df.iloc[idx::len(df)//len(x)]  
            
            if len(time_series) > 0:
                times_numeric = pd.to_numeric(time_series['time'])
                values = time_series['temperature'].values
                new_time = pd.to_numeric(times[time_step])

                try:
                    if technique == 'linear':
                        z_interpolated[i, j] = linear_interpolation(times_numeric, values, [new_time])[0]
                    elif technique == 'spline':
                        z_interpolated[i, j] = spline_interpolation(times_numeric, values, [new_time])[0]
                    elif technique == 'arima':
                        z_interpolated[i, j] = arima_forecast(times_numeric, values, [new_time])[0]
                    elif technique == 'exponential_smoothing':
                        z_interpolated[i, j] = exponential_smoothing(times_numeric, values, [new_time])[0]
                    elif technique == 'prophet':
                        z_interpolated[i, j] = prophet_forecast(times_numeric, values, [new_time])[0]
                    elif technique == 'kalman_filter':
                        z_interpolated[i, j] = kalman_filter(times_numeric, values, [new_time])[0]
                    elif technique == 'advanced_forecast':
                        z_interpolated[i, j] = advanced_forecast(times_numeric, values, [new_time])[0]
                    elif technique == 'gaussian_process':
                        z_interpolated[i, j] = gaussian_process_regression(times_numeric, values, [new_time])[0]
                    elif technique == 'ssa':
                        z_interpolated[i, j] = singular_spectrum_analysis(times_numeric, values, [new_time])[0]
                    elif technique == 'sarima':
                        z_interpolated[i, j] = sarima_forecast(times_numeric, values, [new_time])[0]
                    else:
                        raise ValueError(f"Unknown temporal interpolation technique: {technique}")
                except Exception as e:
                    logger.error(f"Error applying {technique} at point ({i}, {j}): {str(e)}")
                    z_interpolated[i, j] = np.nan
            else:
                z_interpolated[i, j] = np.nan

    plt.figure(figsize=(10, 6))
    plt.contourf(grid_x, grid_y, z_interpolated, cmap='viridis', levels=20)
    plt.colorbar(label='Temperature (°C)')
    plt.title(f'Climate Map at Time Step {time_step} using {technique.capitalize()}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close()

    return Image.open(buf), max_time_step

if __name__ == '__main__':
    app.run(debug=True)