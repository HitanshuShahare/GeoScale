from flask import Flask, request, send_file, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.spatial.distance import cdist
from pykrige.ok import OrdinaryKriging
import io
from flask_cors import CORS
from scipy.interpolate import NearestNDInterpolator, RBFInterpolator
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
CORS(app)

def linear_interpolation(x, y, x_new):
    return np.interp(x_new, x, y)

def cubic_spline_interpolation(x, y, x_new):
    cs = interpolate.CubicSpline(x, y)
    return cs(x_new)

def nearest_neighbor_interpolation(x, y, x_new):
    interp_func = NearestNDInterpolator(x.reshape(-1, 1), y)
    return interp_func(x_new.reshape(-1, 1))

def polynomial_interpolation(x, y, x_new, degree=3):
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X = poly_features.fit_transform(x.reshape(-1, 1))
    model = LinearRegression()
    model.fit(X, y)
    X_new = poly_features.transform(x_new.reshape(-1, 1))
    return model.predict(X_new)

def spline_interpolation(x, y, x_new):
    tck = interpolate.splrep(x, y)
    return interpolate.splev(x_new, tck)

def radial_basis_function_interpolation(x, y, x_new):
    rbf = RBFInterpolator(x.reshape(-1, 1), y, kernel='multiquadric')
    return rbf(x_new.reshape(-1, 1))

def inverse_distance_weighting_interpolation(x, y, x_new):
    y_new = []
    for xi in x_new:
        dist = cdist([[xi]], x[:, None], 'euclidean').flatten()
        weights = 1 / (dist**2 + 1e-10)
        weights[dist == 0] = 1e-10  # Avoid division by zero
        y_new.append(np.sum(weights * y) / np.sum(weights))
    return np.array(y_new)

def kriging_interpolation(x, y, x_new):
    OK = OrdinaryKriging(x, np.zeros_like(x), y, variogram_model='linear')
    y_new, _ = OK.execute('grid', x_new, np.zeros(1))
    return y_new[0]

def natural_neighbor_interpolation(x, y, x_new):
    # Placeholder for natural neighbor interpolation
    return np.interp(x_new, x, y)

def trend_surface_analysis(x, y, x_new, degree=3):
    return polynomial_interpolation(x, y, x_new, degree)

@app.route('/interpolate', methods=['POST'])
def interpolate():
    try:
        file = request.files['file']
        technique = request.form['technique']

        df = pd.read_csv(file)
        x = df['x'].values
        y = df['y'].values

        x_new = np.linspace(x.min(), x.max(), 300)

        interpolation_techniques = {
            'linear': linear_interpolation,
            'cubic_spline': cubic_spline_interpolation,
            'nearest_neighbor': nearest_neighbor_interpolation,
            'polynomial': polynomial_interpolation,
            'spline': spline_interpolation,
            'rbf': radial_basis_function_interpolation,
            'idw': inverse_distance_weighting_interpolation,
            'kriging': kriging_interpolation,
            'natural_neighbor': natural_neighbor_interpolation,
            'trend': trend_surface_analysis
        }

        if technique not in interpolation_techniques:
            return jsonify(error="Invalid interpolation technique"), 400

        if technique == 'polynomial' or technique == 'trend':
            y_new = interpolation_techniques[technique](x, y, x_new, degree=3)
        else:
            y_new = interpolation_techniques[technique](x, y, x_new)

        plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'o', label='Original data')
        plt.plot(x_new, y_new, '-', label=f'{technique.replace("_", " ").title()} interpolation')
        plt.legend()
        plt.title(f'{technique.replace("_", " ").title()} Interpolation')
        plt.xlabel('X')
        plt.ylabel('Y')

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()

        return send_file(buf, mimetype='image/png')
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route('/generate_geostatistical_image', methods=['POST'])
def generate_geostatistical_image():
    try:
        file = request.files['file']

        df = pd.read_csv(file)
        x = df['x'].values
        y = df['y'].values

        grid_x, grid_y = np.meshgrid(np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100))

        OK = OrdinaryKriging(x, np.zeros_like(x), y, variogram_model='linear')
        z, _ = OK.execute('grid', grid_x.flatten(), grid_y.flatten())
        z = z.reshape(grid_x.shape)

        plt.figure(figsize=(10, 6))
        plt.contourf(grid_x, grid_y, z, cmap='viridis')
        plt.colorbar(label='Value')
        plt.scatter(x, y, c=y, cmap='viridis', edgecolor='black', linewidth=0.5)
        plt.title('Geostatistical Image using Kriging Interpolation')
        plt.xlabel('X')
        plt.ylabel('Y')

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()

        return send_file(buf, mimetype='image/png')
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route('/downscaled_geostatistical_image', methods=['POST'])
def downscaled_geostatistical_image():
    try:
        file = request.files['file']
        technique = request.form['technique']
        factor = int(request.form.get('factor', 5))  # Default downscaling factor is 5

        df = pd.read_csv(file)
        x = df['x'].values
        y = df['y'].values

        full_grid_x, full_grid_y = np.meshgrid(np.linspace(x.min(), x.max(), 500),
                                               np.linspace(y.min(), y.max(), 500))

        interpolation_techniques = {
            'linear': linear_interpolation,
            'cubic_spline': cubic_spline_interpolation,
            'nearest_neighbor': nearest_neighbor_interpolation,
            'polynomial': polynomial_interpolation,
            'spline': spline_interpolation,
            'rbf': radial_basis_function_interpolation,
            'idw': inverse_distance_weighting_interpolation,
            'kriging': kriging_interpolation,
            'natural_neighbor': natural_neighbor_interpolation,
            'trend': trend_surface_analysis
        }

        if technique not in interpolation_techniques:
            return jsonify(error="Invalid interpolation technique"), 400

        if technique == 'polynomial' or technique == 'trend':
            full_z = interpolation_techniques[technique](x, y, full_grid_x.flatten(), degree=3)
        else:
            full_z = interpolation_techniques[technique](x, y, full_grid_x.flatten())

        full_z = full_z.reshape(full_grid_x.shape)

        plt.figure(figsize=(10, 6))
        plt.contourf(full_grid_x, full_grid_y, full_z, cmap='viridis', levels=20)
        plt.colorbar(label='Value')
        plt.scatter(x, y, c=y, cmap='viridis', edgecolor='black', linewidth=0.5)
        plt.title(f'Full-resolution Geostatistical Image using {technique.replace("_", " ").title()}')
        plt.xlabel('X')
        plt.ylabel('Y')

        full_buf = io.BytesIO()
        plt.savefig(full_buf, format='png', dpi=300)
        plt.close()

        from PIL import Image

        full_buf.seek(0)
        full_image = Image.open(full_buf)

        width, height = full_image.size
        new_width = width // factor
        new_height = height // factor

        downscaled_image = full_image.resize((new_width, new_height), Image.LANCZOS)

        downscaled_buf = io.BytesIO()
        downscaled_image.save(downscaled_buf, format='png')
        downscaled_buf.seek(0)

        return send_file(downscaled_buf, mimetype='image/png')
    except Exception as e:
        return jsonify(error=str(e)), 500

if __name__ == '__main__':
    app.run(debug=True)
