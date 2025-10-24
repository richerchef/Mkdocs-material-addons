import numpy as np
import plotly.graph_objects as go
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
    coeffs_list = []
    x_fit = []
    y_fit = []
    
    for i in range(0, len(x), chunk_size):
        x_chunk = x[i:i+chunk_size]
        y_chunk = y[i:i+chunk_size]
        
        # Polynomial fit using NumPy
        coeffs = np.polyfit(x_chunk, y_chunk, deg=poly_degree)
        y_pred = np.polyval(coeffs, x_chunk)
        
        coeffs_list.append(coeffs.tolist())
        x_fit.extend(x_chunk)
        y_fit.extend(y_pred)
    
    return np.array(x_fit), np.array(y_fit), coeffs_list

def estimate_size(obj):
    """Rough estimate of serialized object size (bytes)."""
    return sys.getsizeof(pickle.dumps(obj))

# === Run scenarios ===
scenarios = ["smooth_with_spikes", "pure_noise", "spiky_noise", "oscillating_drift"]
poly_degree = 3
chunk_size = 100

for scenario in scenarios:
    print(f"\n=== Scenario: {scenario} ===")
    x, y = simulate_data(scenario)
    
    x_fit, y_fit, coeffs_list = compress_timeseries(x, y, chunk_size, poly_degree)
    
    # Compression stats
    raw_size = estimate_size(y)
    compressed_size = estimate_size(coeffs_list)
    compression_ratio = compressed_size / raw_size
    print(f"Raw size:        {raw_size/1024:.2f} KB")
    print(f"Compressed size: {compressed_size/1024:.2f} KB")
    print(f"Compression ratio: {compression_ratio:.3f} ({(1-compression_ratio)*100:.1f}% smaller)")
    
    # === Plotly visualization ===
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="lines",
        name="Raw Data",
        line=dict(color="royalblue", width=1),
        opacity=0.6
    ))
    fig.add_trace(go.Scatter(
        x=x_fit, y=y_fit,
        mode="lines",
        name=f"Thumbnail (deg={poly_degree})",
        line=dict(color="orange", width=2)
    ))

    fig.update_layout(
        title=f"Mathematical Thumbnail Compression – {scenario.replace('_', ' ').title()}",
        xaxis_title="Time",
        yaxis_title="Value",
        legend=dict(x=0.01, y=0.99),
        template="plotly_white",
        width=900, height=450
    )

    fig.show()
