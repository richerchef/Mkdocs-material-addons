import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import sys
import pickle

def simulate_data(scenario, n_points=2000):
    """Generate different time series shapes."""
    x = np.linspace(0, 100, n_points)
    np.random.seed(42)
    
    if scenario == "smooth_with_spikes":
        y = np.sin(x / 5) * 10 + np.cos(x / 15) * 5 + np.random.normal(0, 0.8, n_points)
        spike_indices = np.random.choice(n_points, 10, replace=False)
        y[spike_indices] += np.random.uniform(10, 25, len(spike_indices))
    elif scenario == "pure_noise":
        y = np.random.normal(0, 5, n_points)
    elif scenario == "spiky_noise":
        y = np.random.normal(0, 1, n_points)
        spike_indices = np.random.choice(n_points, 30, replace=False)
        y[spike_indices] += np.random.uniform(5, 15, len(spike_indices))
    elif scenario == "oscillating_drift":
        y = (np.sin(x / 4) * 8 + (x / 5)) + np.random.normal(0, 1, n_points)
    else:
        raise ValueError("Unknown scenario")
    
    return x, y

def compress_timeseries(x, y, chunk_size=100, poly_degree=3):
    """Fit polynomial models per chunk and reconstruct."""
    models = []
    x_fit = []
    y_fit = []
    
    for i in range(0, len(x), chunk_size):
        x_chunk = x[i:i+chunk_size]
        y_chunk = y[i:i+chunk_size]
        
        poly = PolynomialFeatures(degree=poly_degree)
        X_poly = poly.fit_transform(x_chunk.reshape(-1, 1))
        model = LinearRegression().fit(X_poly, y_chunk)
        
        y_pred = model.predict(X_poly)
        x_fit.extend(x_chunk)
        y_fit.extend(y_pred)
        
        models.append(model.coef_.tolist() + [model.intercept_])
    
    return np.array(x_fit), np.array(y_fit), models

def estimate_size(obj):
    """Rough estimate of memory footprint (bytes)."""
    return sys.getsizeof(pickle.dumps(obj))

# === Run scenarios ===
scenarios = ["smooth_with_spikes", "pure_noise", "spiky_noise", "oscillating_drift"]
poly_degree = 3
chunk_size = 100

for scenario in scenarios:
    print(f"\n=== Scenario: {scenario} ===")
    x, y = simulate_data(scenario)
    
    x_fit, y_fit, models = compress_timeseries(x, y, chunk_size, poly_degree)
    
    # Estimate file sizes
    raw_size = estimate_size(y)
    compressed_size = estimate_size(models)
    compression_ratio = compressed_size / raw_size
    
    print(f"Raw size:        {raw_size/1024:.2f} KB")
    print(f"Compressed size: {compressed_size/1024:.2f} KB")
    print(f"Compression ratio: {compression_ratio:.3f} ({(1-compression_ratio)*100:.1f}% smaller)")
    
    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(x, y, label="Raw data", alpha=0.6)
    plt.plot(x_fit, y_fit, label=f"Thumbnail (deg={poly_degree})", linewidth=2)
    plt.title(f"Mathematical Thumbnail Compression – {scenario.replace('_', ' ').title()}")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()
