import numpy as np
import json
import plotly.graph_objects as go
import time
from pathlib import Path

# ---------- CONFIG ----------
SAVE_PATH = Path("compressed_timeseries.json")
CHUNK_SIZE = 200
MAX_DEGREE = 5
ERROR_THRESHOLD = 0.005  # tune this to control compression quality


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


# ---------- STEP 2: Adaptive Compression ----------
def compress_timeseries_adaptive(x, y, chunk_size=CHUNK_SIZE, max_degree=MAX_DEGREE, error_threshold=ERROR_THRESHOLD):
    compressed = []
    n_chunks = len(x) // chunk_size

    for i in range(n_chunks):
        start, end = i * chunk_size, (i + 1) * chunk_size
        x_chunk, y_chunk = x[start:end], y[start:end]

        best_degree, best_mse, best_coeffs = None, np.inf, None

        # Try multiple polynomial degrees
        for deg in range(1, max_degree + 1):
            coeffs = np.polyfit(x_chunk, y_chunk, deg)
            poly = np.poly1d(coeffs)
            y_fit = poly(x_chunk)
            mse = np.mean((y_fit - y_chunk) ** 2)

            if mse < best_mse:
                best_mse = mse
                best_degree = deg
                best_coeffs = coeffs

            # Stop early if the fit is "good enough"
            if mse < error_threshold:
                break

        # Compute bounds
        poly_best = np.poly1d(best_coeffs)
        y_fit = poly_best(x_chunk)
        residuals = y_chunk - y_fit
        upper = np.max(residuals)
        lower = np.min(residuals)
        min_val = np.min(y_chunk)
        max_val = np.max(y_chunk)

        compressed.append({
            "x_start": float(x_chunk[0]),
            "x_end": float(x_chunk[-1]),
            "coeffs": best_coeffs.tolist(),
            "degree": best_degree,
            "mse": float(best_mse),
            "upper": float(upper),
            "lower": float(lower),
            "min": float(min_val),
            "max": float(max_val)
        })

    return compressed


# ---------- STEP 3: Save and Load JSON ----------
def save_compressed(data, path=SAVE_PATH):
    with open(path, "w") as f:
        json.dump(data, f)


def load_compressed(path=SAVE_PATH):
    with open(path, "r") as f:
        return json.load(f)


# ---------- STEP 4: Decompress ----------
def decompress_timeseries(compressed_data, n_points_per_chunk=CHUNK_SIZE):
    x_all, y_all, upper_all, lower_all = [], [], [], []
    for chunk in compressed_data:
        x_chunk = np.linspace(chunk["x_start"], chunk["x_end"], n_points_per_chunk)
        poly = np.poly1d(chunk["coeffs"])
        y_fit = poly(x_chunk)
        y_upper = y_fit + chunk["upper"]
        y_lower = y_fit + chunk["lower"]

        x_all.extend(x_chunk)
        y_all.extend(y_fit)
        upper_all.extend(y_upper)
        lower_all.extend(y_lower)

    return np.array(x_all), np.array(y_all), np.array(upper_all), np.array(lower_all)


# ---------- STEP 5: Plot + Timing ----------
def plot_comparison(x_raw, y_raw, compressed_data, scenario_name="Scenario"):
    start = time.time()
    x_c, y_c, y_upper, y_lower = decompress_timeseries(compressed_data)
    compressed_time = time.time() - start

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_raw, y=y_raw, mode='lines', name='Raw Data', line=dict(width=1)))
    fig.add_trace(go.Scatter(x=x_c, y=y_c, mode='lines', name='Compressed Fit', line=dict(width=2)))
    fig.add_trace(go.Scatter(x=x_c, y=y_upper, mode='lines', name='Upper Bound', line=dict(dash='dot')))
    fig.add_trace(go.Scatter(x=x_c, y=y_lower, mode='lines', name='Lower Bound', line=dict(dash='dot')))

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
    compressed = compress_timeseries_adaptive(x, y)
    save_compressed(compressed)
    loaded = load_compressed()
    plot_comparison(x, y, loaded, scenario)
