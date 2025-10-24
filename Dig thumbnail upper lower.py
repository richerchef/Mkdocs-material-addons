import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# --- Helper: generate sample scenarios ---
def generate_scenarios(n_points=5000):
    x = np.linspace(0, 20, n_points)
    scenarios = {}

    scenarios["Smooth sine"] = np.sin(x)
    scenarios["Noisy sine"] = np.sin(x) + np.random.normal(0, 0.2, n_points)
    scenarios["Drift + noise"] = 0.2*x + np.sin(x) + np.random.normal(0, 0.3, n_points)
    scenarios["Pure noise (no baseline)"] = np.random.normal(0, 0.5, n_points)
    scenarios["Spike anomalies"] = np.sin(x)
    spike_idx = np.random.choice(n_points, 30, replace=False)
    scenarios["Spike anomalies"][spike_idx] += np.random.normal(2, 0.5, 30)
    return x, scenarios


# --- Chunked compression using np.polyfit ---
def chunked_polyfit(x, y, chunk_size=1000, degree=5):
    n_chunks = len(x) // chunk_size
    fits = []
    for i in range(n_chunks):
        start, end = i * chunk_size, (i + 1) * chunk_size
        x_chunk = x[start:end]
        y_chunk = y[start:end]

        coeffs = np.polyfit(x_chunk, y_chunk, degree)
        y_fit = np.polyval(coeffs, x_chunk)
        residuals = y_chunk - y_fit

        sigma = np.std(residuals)
        y_upper = y_fit + sigma
        y_lower = y_fit - sigma

        # Outer bounds (based on actual max/min residuals)
        y_outer_max = y_fit + np.max(residuals)
        y_outer_min = y_fit + np.min(residuals)

        fits.append({
            "x": x_chunk,
            "y_fit": y_fit,
            "y_upper": y_upper,
            "y_lower": y_lower,
            "y_outer_max": y_outer_max,
            "y_outer_min": y_outer_min,
            "coeffs": coeffs
        })

    total_coeffs = n_chunks * (degree + 1)
    ratio = total_coeffs / len(y)
    return fits, ratio


# --- Generate and plot ---
x, scenarios = generate_scenarios()
rows = len(scenarios)
fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                    subplot_titles=list(scenarios.keys()))

# Measure timing
timing_results = {}

for row, (name, y) in enumerate(scenarios.items(), start=1):
    # Raw plot timing
    start_raw = time.perf_counter()
    raw_trace = go.Scatter(x=x, y=y, mode='lines', line=dict(color='lightgray'))
    end_raw = time.perf_counter()
    raw_time = end_raw - start_raw

    # Compressed timing
    start_comp = time.perf_counter()
    fits, ratio = chunked_polyfit(x, y, chunk_size=1000, degree=5)
    comp_traces = []
    for f in fits:
        comp_traces.append(go.Scatter(
            x=f["x"], y=f["y_fit"], mode='lines', line=dict(color='blue', width=2)))
        # sigma bounds
        comp_traces.append(go.Scatter(
            x=np.concatenate([f["x"], f["x"][::-1]]),
            y=np.concatenate([f["y_upper"], f["y_lower"][::-1]]),
            fill='toself', fillcolor='rgba(0,0,255,0.1)',
            line=dict(color='rgba(255,255,255,0)')))
        # max/min bounds
        comp_traces.append(go.Scatter(
            x=np.concatenate([f["x"], f["x"][::-1]]),
            y=np.concatenate([f["y_outer_max"], f["y_outer_min"][::-1]]),
            fill='toself', fillcolor='rgba(255,0,0,0.1)',
            line=dict(color='rgba(255,255,255,0)')))
    end_comp = time.perf_counter()
    comp_time = end_comp - start_comp

    # Add traces to figure
    fig.add_trace(raw_trace, row=row, col=1)
    for t in comp_traces:
        fig.add_trace(t, row=row, col=1)

    timing_results[name] = {
        "compression_ratio": ratio,
        "raw_time": raw_time,
        "compressed_time": comp_time
    }

    fig.update_yaxes(title_text=name, row=row, col=1)

# --- Layout ---
fig.update_layout(
    height=300 * rows,
    title="Chunked Polynomial Compression with σ + Max/Min Bounds",
    showlegend=False,
    template="plotly_white"
)

fig.show()

# --- Print timing + compression stats ---
print("=== Compression and Timing Results ===")
for name, stats in timing_results.items():
    print(f"{name}:")
    print(f"  Compression ratio: {stats['compression_ratio']:.5f}")
    print(f"  Plot (raw data): {stats['raw_time']*1000:.2f} ms")
    print(f"  Plot (compressed): {stats['compressed_time']*1000:.2f} ms")
    print()
