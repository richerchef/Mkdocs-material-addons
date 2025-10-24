import numpy as np
import json
import plotly.graph_objects as go
import time
from pathlib import Path

# ---------- CONFIG ----------
SAVE_PATH = Path("compressed_timeseries.json")
CHUNK_SIZE = 200
MAX_DEGREE = 5
ERROR_THRESHOLD = 0.005  # tune this for quality


# ---------- STEP 1: Generate Mock Data ----------
def generate_timeseries(n_points=5000, scenario="normal"):
    x = np.linspace(0, 100, n_points)
    if scenario == "normal":
        y = np.sin(x / 5) + np.random.normal(0, 0.1, n_points)
    elif scenario == "noisy":
        y = np.sin(x / 5) + np.random.normal(0, 0.3, n_points)
    elif scenario == "trend":
        y = 0.02 * x + np.sin(x / 4) + np.random.normal(0, 0.05, n_points)
    elif scenario == "spikes":
        y = np.sin(x / 5)
        spikes = np.random.choice(n_points, size=30, replace=False)
        y[spikes] += np.random.uniform(2, 4, size=len(spikes))
    elif scenario == "flat":
        y = np.random.normal(0, 0.05, n_points)
    else:
        raise ValueError("Unknown scenario")
    return x, y


# ---------- STEP 2: Adaptive Compression with sigma bounds ----------
def compress_timeseries_adaptive_sigma(x, y, chunk_size=CHUNK_SIZE, max_degree=MAX_DEGREE, error_threshold=ERROR_THRESHOLD):
    compressed = []
    n_chunks = len(x) // chunk_size

    for i in range(n_chunks):
        start, end = i * chunk_size, (i + 1) * chunk_size
        x_chunk, y_chunk = x[start:end], y[start:end]

        best_degree, best_mse, best_coeffs = None, np.inf, None

        for deg in range(1, max_degree + 1):
            coeffs = np.polyfit(x_chunk, y_chunk, deg)
            poly = np.poly1d(coeffs)
            y_fit = poly(x_chunk)
            mse = np.mean((y_fit - y_chunk) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_degree = deg
                best_coeffs = coeffs
            if mse < error_threshold:
                break

        poly_best = np.poly1d(best_coeffs)
        y_fit = poly_best(x_chunk)
        residuals = y_chunk - y_fit
        sigma = np.std(residuals)
        upper_sigma = y_fit + sigma
        lower_sigma = y_fit - sigma

        # max/min residual bounds
        upper_max = y_fit + np.max(residuals)
        lower_min = y_fit + np.min(residuals)

        compressed.append({
            "x_start": float(x_chunk[0]),
            "x_end": float(x_chunk[-1]),
            "coeffs": best_coeffs.tolist(),
            "degree": best_degree,
            "mse": float(best_mse),
            "sigma": float(sigma),
            "upper_sigma": float(np.max(upper_sigma - y_fit)),  # store relative
            "lower_sigma": float(np.min(lower_sigma - y_fit)),   # store relative
            "upper_max": float(np.max(upper_max - y_fit)),       # relative
            "lower_min": float(np.min(lower_min - y_fit)),       # relative
            "min": float(np.min(y_chunk)),
            "max": float(np.max(y_chunk))
        })
    return compressed


# ---------- STEP 3: Save / Load JSON ----------
def save_compressed(data, path=SAVE_PATH):
    with open(path, "w") as f:
        json.dump(data, f)


def load_compressed(path=SAVE_PATH):
    with open(path, "r") as f:
        return json.load(f)


# ---------- STEP 4: Decompress ----------
def decompress_timeseries_sigma(compressed_data, n_points_per_chunk=CHUNK_SIZE):
    x_all, y_all, upper_sigma_all, lower_sigma_all, upper_max_all, lower_min_all = [], [], [], [], [], []
    for chunk in compressed_data:
        x_chunk = np.linspace(chunk["x_start"], chunk["x_end"], n_points_per_chunk)
        poly = np.poly1d(chunk["coeffs"])
        y_fit = poly(x_chunk)

        upper_sigma = y_fit + chunk["upper_sigma"]
        lower_sigma = y_fit + chunk["lower_sigma"]
        upper_max = y_fit + chunk["upper_max"]
        lower_min = y_fit + chunk["lower_min"]

        x_all.extend(x_chunk)
        y_all.extend(y_fit)
        upper_sigma_all.extend(upper_sigma)
        lower_sigma_all.extend(lower_sigma)
        upper_max_all.extend(upper_max)
        lower_min_all.extend(lower_min)

    return (np.array(x_all), np.array(y_all),
            np.array(upper_sigma_all), np.array(lower_sigma_all),
            np.array(upper_max_all), np.array(lower_min_all))


# ---------- STEP 5: Plot + Timing ----------
def plot_comparison_sigma(x_raw, y_raw, compressed_data, scenario_name="Scenario"):
    start = time.time()
    x_c, y_c, y_upper_sigma, y_lower_sigma, y_upper_max, y_lower_min = decompress_timeseries_sigma(compressed_data)
    compressed_time = time.time() - start

    fig = go.Figure()
    # Raw data
    fig.add_trace(go.Scatter(x=x_raw, y=y_raw, mode='lines', name='Raw', line=dict(color='lightgray')))
    # Compressed fit
    fig.add_trace(go.Scatter(x=x_c, y=y_c, mode='lines', name='Fit', line=dict(color='blue', width=2)))
    # Sigma bounds
    fig.add_trace(go.Scatter(x=x_c, y=y_upper_sigma, mode='lines', name='Sigma Upper', line=dict(dash='dot'), line_color='cyan'))
    fig.add_trace(go.Scatter(x=x_c, y=y_lower_sigma, mode='lines', name='Sigma Lower', line=dict(dash='dot'), line_color='cyan'))
    # Max/min bounds
    fig.add_trace(go.Scatter(x=x_c, y=y_upper_max, mode='lines', name='Max Upper', fill=None, line=dict(color='red', dash='dot')))
    fig.add_trace(go.Scatter(x=x_c, y=y_lower_min, mode='lines', name='Min Lower', fill='tonexty', fillcolor='rgba(255,0,0,0.1)', line=dict(color='red', dash='dot')))

    fig.update_layout(title=f"{scenario_name} | Compressed Plot Load: {compressed_time:.4f}s",
                      xaxis_title="X", yaxis_title="Y")
    fig.show()

    raw_size = (len(x_raw) * 8 + len(y_raw) * 8) / 1024
    comp_size = Path(SAVE_PATH).stat().st_size / 1024
    print(f"Raw size: {raw_size:.2f} KB | Compressed JSON size: {comp_size:.2f} KB")
    avg_deg = np.mean([c["degree"] for c in compressed_data])
    print(f"Average polynomial degree used: {avg_deg:.2f}")


# ---------- STEP 6: Run Scenarios ----------
for scenario in ["normal", "noisy", "trend", "spikes", "flat"]:
    print(f"\n--- Running {scenario} scenario ---")
    x, y = generate_timeseries(scenario=scenario)
    compressed = compress_timeseries_adaptive_sigma(x, y)
    save_compressed(compressed)
    loaded = load_compressed()
    plot_comparison_sigma(x, y, loaded, scenario)
