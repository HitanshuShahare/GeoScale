from flask import Flask, request, send_file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.spatial.distance import cdist
from pykrige.ok import OrdinaryKriging
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def linear_interpolation(x, y, x_new):
    return np.interp(x_new, x, y)

def cubic_spline_interpolation(x, y, x_new):
    cs = interpolate.CubicSpline(x, y)
    return cs(x_new)

def nearest_neighbor_interpolation(x, y, x_new):
    interp_func = interpolate.interp1d(x, y, kind='nearest')
    return interp_func(x_new)

def polynomial_interpolation(x, y, x_new):
    poly = np.polyfit(x, y, 3)  # 3rd degree polynomial
    return np.polyval(poly, x_new)

def radial_basis_function_interpolation(x, y, x_new):
    rbf = interpolate.Rbf(x, y, function='multiquadric')
    return rbf(x_new)

def kriging_interpolation(x, y, x_new):
    OK = OrdinaryKriging(x, np.zeros_like(x), y, variogram_model='linear')
    y_new, _ = OK.execute('grid', x_new, np.zeros(1))
    return y_new[0]

def inverse_distance_weighting_interpolation(x, y, x_new):
    y_new = []
    for xi in x_new:
        dist = cdist([[xi]], x[:, None], 'euclidean').flatten()
        weights = 1 / dist
        weights[dist == 0] = 1e-10  # Avoid division by zero
        y_new.append(np.sum(weights * y) / np.sum(weights))
    return np.array(y_new)

@app.route('/interpolate', methods=['POST'])
def interpolate():
    # Get the CSV file and interpolation technique from the request
    file = request.files['file']
    technique = request.form['technique']

    # Read the CSV file
    df = pd.read_csv(file)
    x = df['x'].values
    y = df['y'].values

    # Create a finer x scale for interpolation
    x_new = np.linspace(x.min(), x.max(), 300)

    # Perform interpolation based on the chosen technique
    if technique == 'linear':
        y_new = linear_interpolation(x, y, x_new)
    elif technique == 'cubic_spline':
        y_new = cubic_spline_interpolation(x, y, x_new)
    elif technique == 'nearest_neighbor':
        y_new = nearest_neighbor_interpolation(x, y, x_new)
    elif technique == 'polynomial':
        y_new = polynomial_interpolation(x, y, x_new)
    elif technique == 'rbf':
        y_new = radial_basis_function_interpolation(x, y, x_new)
    elif technique == 'kriging':
        y_new = kriging_interpolation(x, y, x_new)
    elif technique == 'idw':
        y_new = inverse_distance_weighting_interpolation(x, y, x_new)
    else:
        return "Invalid interpolation technique", 400

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'o', label='Original data')
    plt.plot(x_new, y_new, '-', label=f'{technique.replace("_", " ").title()} interpolation')
    plt.legend()
    plt.title(f'{technique.replace("_", " ").title()} Interpolation')
    plt.xlabel('X')
    plt.ylabel('Y')

    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    # Return the image
    return send_file(buf, mimetype='image/png')

@app.route('/generate_geostatistical_image', methods=['POST'])
def generate_geostatistical_image():
    # Get the CSV file from the request
    file = request.files['file']

    # Read the CSV file
    df = pd.read_csv(file)
    x = df['x'].values
    y = df['y'].values

    # Create a grid for the geostatistical image
    grid_x, grid_y = np.meshgrid(np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100))

    # Perform kriging interpolation
    OK = OrdinaryKriging(x, np.zeros_like(x), y, variogram_model='linear')
    z, _ = OK.execute('grid', grid_x[0], grid_y[:, 0])

    # Create the geostatistical image
    plt.figure(figsize=(10, 6))
    plt.contourf(grid_x, grid_y, z, cmap='viridis')
    plt.colorbar(label='Value')
    plt.title('Geostatistical Image using Kriging Interpolation')
    plt.xlabel('X')
    plt.ylabel('Y')

    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    # Return the image
    return send_file(buf, mimetype='image/png')

@app.route('/downscaled_geostatistical_image', methods=['POST'])
def downscaled_geostatistical_image():
    # Get the CSV file from the request
    file = request.files['file']

    # Define the downscaling factor
    factor = 5  # You can adjust this value as needed

    # Read the CSV file
    df = pd.read_csv(file)
    x = df['x'].values
    y = df['y'].values

    # Create a grid for the full-resolution geostatistical image
    full_grid_x, full_grid_y = np.meshgrid(np.linspace(x.min(), x.max(), 500), 
                                           np.linspace(y.min(), y.max(), 500))

    # Perform kriging interpolation on the full data
    OK = OrdinaryKriging(x, np.zeros_like(x), y, variogram_model='linear')
    full_z, _ = OK.execute('grid', full_grid_x[0], full_grid_y[:, 0])

    # Create the full-resolution geostatistical image
    plt.figure(figsize=(10, 6))
    plt.contourf(full_grid_x, full_grid_y, full_z, cmap='viridis', levels=20)
    plt.colorbar(label='Value')
    plt.title(f'Full-resolution Geostatistical Image using Kriging')
    plt.xlabel('X')
    plt.ylabel('Y')

    # Save the full-resolution plot to a bytes buffer
    full_buf = io.BytesIO()
    plt.savefig(full_buf, format='png', dpi=300)
    plt.close()

    # Create the downscaled image
    from PIL import Image

    # Open the full-resolution image
    full_buf.seek(0)
    full_image = Image.open(full_buf)

    # Calculate new dimensions
    width, height = full_image.size
    new_width = width // factor
    new_height = height // factor

    # Resize the image
    downscaled_image = full_image.resize((new_width, new_height), Image.LANCZOS)

    # Save the downscaled image to a new bytes buffer
    downscaled_buf = io.BytesIO()
    downscaled_image.save(downscaled_buf, format='png')
    downscaled_buf.seek(0)

    # Return the downscaled image
    return send_file(downscaled_buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)