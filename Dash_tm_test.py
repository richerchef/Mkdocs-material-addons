import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

# -----------------------------
# Sample dataset
# -----------------------------
data = {
    'Department': ['IT', 'IT', 'HR', 'HR', 'Finance', 'Finance'],
    'Team': ['Infrastructure', 'Software', 'Recruitment', 'Training', 'Accounts', 'Audit'],
    'Project': ['Servers', 'App Dev', 'Hiring', 'Onboarding', 'Budget', 'Compliance'],
    'Duration': [50, 10, 30, 15, 25, 20],
    'Completion': ['Complete', 'In Progress', 'Deferred', 'Complete', 'Partial', 'Complete']
}
df = pd.DataFrame(data)

# -----------------------------
# Treemap generator
# -----------------------------
def make_treemap(dataframe):
    fig = px.treemap(
        dataframe,
        path=['Department', 'Team', 'Project'],
        values='Duration',
        color='Completion',
        color_discrete_map={
            'Complete': 'green',
            'In Progress': 'orange',
            'Deferred': 'red',
            'Partial': 'yellow'
        }
    )
    fig.update_layout(margin=dict(t=40, l=0, r=0, b=0))
    return fig

# -----------------------------
# App setup
# -----------------------------
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("Project Overview Treemap"),
    dcc.Graph(id='treemap', figure=make_treemap(df)),

    html.H3("Project Data Table"),
    dash_table.DataTable(
        id='data-table',
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict('records'),
        filter_action="native",
        sort_action="native",
        page_action="native",
        page_size=5,
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left', 'padding': '8px'},
        style_header={'fontWeight': 'bold', 'backgroundColor': '#f0f0f0'}
    ),
    html.Div(id='debug', style={'marginTop': '20px', 'fontSize': '12px', 'color': 'gray'})
])

# -----------------------------
# Callback: link treemap click → table filter
# -----------------------------
@app.callback(
    [Output('data-table', 'data'),
     Output('debug', 'children')],
    [Input('treemap', 'clickData')]
)
def update_table(clickData):
    # If nothing clicked, show all data
    if clickData is None:
        return df.to_dict('records'), "No selection"

    # The label clicked is in clickData['points'][0]['label']
    label = clickData['points'][0]['label']

    # Filter rows where any hierarchy column matches the clicked label
    filtered = df[
        (df['Department'] == label) |
        (df['Team'] == label) |
        (df['Project'] == label)
    ]

    # If no match (e.g., clicked background), return all
    if filtered.empty:
        return df.to_dict('records'), f"No match for '{label}'"

    return filtered.to_dict('records'), f"Filtered by: {label}"

# -----------------------------
# Run
# -----------------------------
if __name__ == '__main__':
    app.run_server(debug=True)
