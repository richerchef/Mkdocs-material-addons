import numpy as np
import plotly.graph_objects as go

# --- Helper function ---
def generate_scenarios(n_points=1000):
    x = np.linspace(0, 10, n_points)
    scenarios = {}

    scenarios["Smooth sine"] = np.sin(x)
    scenarios["Noisy sine"] = np.sin(x) + np.random.normal(0, 0.2, n_points)
    scenarios["Drift + noise"] = 0.2*x + np.sin(x) + np.random.normal(0, 0.3, n_points)
    scenarios["Pure noise (no baseline)"] = np.random.normal(0, 0.5, n_points)
    scenarios["Spike anomalies"] = np.sin(x)
    spike_idx = np.random.choice(n_points, 10, replace=False)
    scenarios["Spike anomalies"][spike_idx] += np.random.normal(2, 0.5, 10)

    return x, scenarios


# --- Compression function ---
def compress_with_polyfit(x, y, degree=5):
    coeffs = np.polyfit(x, y, degree)
    y_fit = np.polyval(coeffs, x)
    residuals = y - y_fit
    sigma = np.std(residuals)
    y_upper = y_fit + sigma
    y_lower = y_fit - sigma

    compression_ratio = len(coeffs) / len(y)
    return y_fit, y_upper, y_lower, coeffs, compression_ratio


# --- Main ---
x, scenarios = generate_scenarios()
fig = go.Figure()

buttons = []
first = True

for i, (name, y) in enumerate(scenarios.items()):
    y_fit, y_upper, y_lower, coeffs, ratio = compress_with_polyfit(x, y, degree=5)

    # Create traces (each hidden by default)
    visible = True if first else False
    first = False

    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=f'{name} - raw',
                             line=dict(color='lightgray'), visible=visible))
    fig.add_trace(go.Scatter(x=x, y=y_fit, mode='lines', name=f'{name} - fit',
                             line=dict(color='blue'), visible=visible))
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([y_upper, y_lower[::-1]]),
        fill='toself', fillcolor='rgba(0,0,255,0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        name=f'{name} - bounds', visible=visible
    ))

    # Dropdown button
    n_traces = 3 * i
    button = dict(
        label=f"{name} (ratio={ratio:.5f})",
        method="update",
        args=[
            {"visible": [False] * len(scenarios) * 3},
            {"title": f"{name} — Polynomial compression (degree 5)<br>"
                      f"Raw points: {len(y)}, Coeffs: {len(coeffs)}, "
                      f"Compression ratio: {ratio:.5f}"}
        ]
    )
    # Make only current scenario visible
    button["args"][0]["visible"][n_traces:n_traces+3] = [True, True, True]
    buttons.append(button)

# --- Layout ---
fig.update_layout(
    title="Mathematical Thumbnails with Upper/Lower Bounds",
    updatemenus=[{
        "buttons": buttons,
        "direction": "down",
        "showactive": True,
        "x": 1.05,
        "xanchor": "left",
        "y": 0.5,
        "yanchor": "middle"
    }],
    xaxis_title="Time",
    yaxis_title="Value",
    template="plotly_white",
    height=600
)

fig.show()
