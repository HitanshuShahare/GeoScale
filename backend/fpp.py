from flask import Flask, request, send_file, jsonify
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.spatial.distance import cdist
from pykrige.ok import OrdinaryKriging
import io
from flask_cors import CORS
from PIL import Image
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
CORS(app)

# Spatial Interpolation Methods
def idw_interpolation(x, y, z, grid_x, grid_y):
    points = np.column_stack((x, y))
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    distances = cdist(grid_points, points)
    weights = 1.0 / (distances**2 + 1e-10)
    weights /= weights.sum(axis=1, keepdims=True)
    return np.dot(weights, z).reshape(grid_x.shape)

def kriging_interpolation(x, y, z, grid_x, grid_y):
    OK = OrdinaryKriging(x, y, z, variogram_model='linear')
    z_interpolated, _ = OK.execute('grid', grid_x.ravel(), grid_y.ravel())
    return z_interpolated.reshape(grid_x.shape)

def rbf_interpolation(x, y, z, grid_x, grid_y):
    rbf = interpolate.Rbf(x, y, z, function='multiquadric')
    return rbf(grid_x, grid_y)

def natural_neighbor_interpolation(x, y, z, grid_x, grid_y):
    from scipy.interpolate import griddata
    return griddata((x, y), z, (grid_x, grid_y), method='cubic')

def spline_interpolation(x, y, z, grid_x, grid_y):
    tck = interpolate.bisplrep(x, y, z)
    return interpolate.bisplev(grid_x[:,0], grid_y[0,:], tck)

def trend_surface_analysis(x, y, z, grid_x, grid_y):
    A = np.column_stack((np.ones_like(x), x, y, x**2, x*y, y**2))
    coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
    X, Y = grid_x, grid_y
    return coeffs[0] + coeffs[1]*X + coeffs[2]*Y + coeffs[3]*X**2 + coeffs[4]*X*Y + coeffs[5]*Y**2

def nearest_neighbor_interpolation(x, y, z, grid_x, grid_y):
    from scipy.interpolate import griddata
    return griddata((x, y), z, (grid_x, grid_y), method='nearest')

# Temporal Interpolation Methods
def linear_interpolation(time, value, time_new):
    return np.interp(time_new, time, value)

def polynomial_interpolation(time, value, time_new):
    poly = np.polyfit(time, value, 3)
    return np.polyval(poly, time_new)

def spline_temporal_interpolation(time, value, time_new):
    cs = interpolate.CubicSpline(time, value)
    return cs(time_new)

def kalman_filter(time, value, time_new):
    x = value[0]
    P = 1.0
    Q = 1e-5
    R = 0.1**2
    result = []
    for t in time_new:
        if t in time:
            z = value[np.where(time == t)[0][0]]
            K = P / (P + R)
            x = x + K * (z - x)
            P = (1 - K) * P
        x = x
        P = P + Q
        result.append(x)
    return np.array(result)

def arima_model(time, value, time_new):
    X = value[:-1].reshape(-1, 1)
    y = value[1:]
    model = LinearRegression().fit(X, y)
    
    forecasts = []
    last_value = value[-1]
    for _ in range(len(time_new)):
        forecast = model.predict([[last_value]])[0]
        forecasts.append(forecast)
        last_value = forecast
    
    return np.array(forecasts)

def exponential_smoothing(time, value, time_new):
    alpha = 0.3
    s = [value[0]]
    for i in range(1, len(value)):
        s.append(alpha * value[i] + (1 - alpha) * s[i-1])
    
    forecasts = []
    last_value = s[-1]
    for _ in range(len(time_new)):
        forecasts.append(last_value)
    
    return np.array(forecasts)

def gaussian_process_regression(time, value, time_new):
    kernel = RBF(length_scale=1.0)
    gpr = GaussianProcessRegressor(kernel=kernel)
    gpr.fit(time.reshape(-1, 1), value)
    return gpr.predict(time_new.reshape(-1, 1))

spatial_interpolation_methods = {
    'idw': idw_interpolation,
    'kriging': kriging_interpolation,
    'natural_neighbor': natural_neighbor_interpolation,
    'spline': spline_interpolation,
    'trend': trend_surface_analysis,
    'rbf': rbf_interpolation,
    'nn': nearest_neighbor_interpolation
}

temporal_interpolation_methods = {
    'linear': linear_interpolation,
    'polynomial': polynomial_interpolation,
    'spline': spline_temporal_interpolation,
    'kalman': kalman_filter,
    'arima': arima_model,
    'exponential_smoothing': exponential_smoothing,
    'gaussian_process': gaussian_process_regression
}

@app.route('/interpolate', methods=['POST'])
def interpolate():
    try:
        file = request.files['file']
        interpolation_type = request.form['interpolationType']
        technique = request.form['technique']
        
        df = pd.read_csv(file)
        
        if interpolation_type == 'spatial':
            x = df['x'].values
            y = df['y'].values
            z = df['z'].values
            
            grid_x, grid_y = np.meshgrid(np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100))
            
            interpolated_z = spatial_interpolation_methods[technique](x, y, z, grid_x, grid_y)
            
            plt.figure(figsize=(10, 6))
            plt.contourf(grid_x, grid_y, interpolated_z, cmap='viridis')
            plt.colorbar(label='Value')
            plt.title(f'{technique.replace("_", " ").title()} Interpolation')
            plt.xlabel('X')
            plt.ylabel('Y')
            
        elif interpolation_type == 'temporal':
            time = df['time'].values
            value = df['value'].values
            time_new = np.linspace(time.min(), time.max(), 100)
            
            interpolated_value = temporal_interpolation_methods[technique](time, value, time_new)
            
            plt.figure(figsize=(10, 6))
            plt.plot(time, value, 'o', label='Original data')
            plt.plot(time_new, interpolated_value, '-', label=f'{technique.replace("_", " ").title()} interpolation')
            plt.legend()
            plt.title(f'{technique.replace("_", " ").title()} Interpolation')
            plt.xlabel('Time')
            plt.ylabel('Value')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        return send_file(buf, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/generate_geostatistical_image', methods=['POST'])
def generate_geostatistical_image():
    try:
        file = request.files['file']
        technique = request.form['technique']
        
        df = pd.read_csv(file)
        x = df['x'].values
        y = df['y'].values
        z = df['z'].values
        
        grid_x, grid_y = np.meshgrid(np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100))
        
        interpolated_z = spatial_interpolation_methods[technique](x, y, z, grid_x, grid_y)
        
        plt.figure(figsize=(10, 6))
        plt.contourf(grid_x, grid_y, interpolated_z, cmap='viridis')
        plt.colorbar(label='Value')
        plt.title(f'Geostatistical Image using {technique.replace("_", " ").title()} Interpolation')
        plt.xlabel('X')
        plt.ylabel('Y')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        return send_file(buf, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/downscaled_geostatistical_image', methods=['POST'])
def downscaled_geostatistical_image():
    try:
        file = request.files['file']
        technique = request.form['technique']
        
        df = pd.read_csv(file)
        x = df['x'].values
        y = df['y'].values
        z = df['z'].values
        
        grid_x, grid_y = np.meshgrid(np.linspace(x.min(), x.max(), 50), np.linspace(y.min(), y.max(), 50))
        
        interpolated_z = spatial_interpolation_methods[technique](x, y, z, grid_x, grid_y)
        
        plt.figure(figsize=(10, 6))
        plt.contourf(grid_x, grid_y, interpolated_z, cmap='viridis')
        plt.colorbar(label='Value')
        plt.title(f'Downscaled Geostatistical Image using {technique.replace("_", " ").title()} Interpolation')
        plt.xlabel('X')
        plt.ylabel('Y')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        img = Image.open(buf)
        downscaled_img = img.resize((50, 50), Image.ANTIALIAS)
        
        downscaled_buf = io.BytesIO()
        downscaled_img.save(downscaled_buf, format='png')
        downscaled_buf.seek(0)
        
        return send_file(downscaled_buf, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
