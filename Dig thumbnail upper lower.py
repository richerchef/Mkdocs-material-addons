import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Helper: generate sample scenarios ---
def generate_scenarios(n_points=2000):
    x = np.linspace(0, 20, n_points)
    scenarios = {}

    scenarios["Smooth sine"] = np.sin(x)
    scenarios["Noisy sine"] = np.sin(x) + np.random.normal(0, 0.2, n_points)
    scenarios["Drift + noise"] = 0.2*x + np.sin(x) + np.random.normal(0, 0.3, n_points)
    scenarios["Pure noise (no baseline)"] = np.random.normal(0, 0.5, n_points)
    scenarios["Spike anomalies"] = np.sin(x)
    spike_idx = np.random.choice(n_points, 20, replace=False)
    scenarios["Spike anomalies"][spike_idx] += np.random.normal(2, 0.5, 20)
    return x, scenarios


# --- Chunked compression with np.polyfit ---
def chunked_polyfit(x, y, chunk_size=500, degree=5):
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

        fits.append({
            "x": x_chunk,
            "y_fit": y_fit,
            "y_upper": y_upper,
            "y_lower": y_lower,
            "coeffs": coeffs
        })

    # compression ratio = total coefficients / total points
    total_coeffs = n_chunks * (degree + 1)
    ratio = total_coeffs / len(y)
    return fits, ratio


# --- Main plotting ---
x, scenarios = generate_scenarios()
rows = len(scenarios)
fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                    subplot_titles=list(scenarios.keys()))

for row, (name, y) in enumerate(scenarios.items(), start=1):
    fits, ratio = chunked_polyfit(x, y, chunk_size=500, degree=5)

    # Raw data (light gray)
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='lines', name=f'{name} raw',
        line=dict(color='lightgray')
    ), row=row, col=1)

    # Add each chunk fit and its bounds
    for f in fits:
        fig.add_trace(go.Scatter(
            x=f["x"], y=f["y_fit"], mode='lines', name=f'{name} fit',
            line=dict(color='blue', width=2)
        ), row=row, col=1)

        fig.add_trace(go.Scatter(
            x=np.concatenate([f["x"], f["x"][::-1]]),
            y=np.concatenate([f["y_upper"], f["y_lower"][::-1]]),
            fill='toself', fillcolor='rgba(0,0,255,0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            name=f'{name} bounds'
        ), row=row, col=1)

    fig.update_yaxes(title_text=name, row=row, col=1)

# --- Layout ---
fig.update_layout(
    height=300 * rows,
    title="Chunked Polynomial Compression with Upper/Lower Bounds",
    showlegend=False,
    template="plotly_white"
)

fig.show()
