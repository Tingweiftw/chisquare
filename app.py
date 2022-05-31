import dash
from dash import dash_table
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import random
import os
import io
import base64
import random

import flask
import pandas as pd
import numpy as np
import scipy.stats as st

from content import DISTRIBUTIONS, DISTRIBUTION_OPTIONS, DESCRIPTION

server = flask.Flask('app')

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/hello-world-stock.csv')

app = dash.Dash(
    'app', 
    server=server,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
   
)

# DATA PROCESSING UTILITY FUNCTIONS
def significant_figure(x):
        return round(x, 4 - int(np.floor(np.log10(abs(x)))))

def parse_contents(contents, filename):
    if contents != None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            if 'csv' in filename:
                # Assume that the user uploaded a CSV file
                df = pd.read_csv(
                    io.StringIO(decoded.decode('utf-8')),
                    header=None)
            elif 'xls' in filename:
                # Assume that the user uploaded an excel file
                df = pd.read_excel(
                    io.BytesIO(decoded),
                    header=None)
        except Exception as e:
            print(e)
            return "error"
        return df
    else:
        return "empty"

def get_best_dist(data,candidate_distr,bin_size,sample_size):
    print(len(data),sample_size)
    data_sample = random.sample(list(data),sample_size)
    dist_results = []
    params = {}
    for dist_name in candidate_distr:
        dist = getattr(st, dist_name)
        param = dist.fit(data_sample)
        params[dist_name] = param
    dist_results = []
    for i in params.items():
        dist_name, param = i[0], i[1]
        dist = getattr(st,dist_name)
        if param !="Error":
            arg = param[:-2]
            loc = param[-2]
            scale = param[-1]
            expected_frequency = [1/bin_size * sample_size] * bin_size
            cumulative_probability = np.linspace(0,1,bin_size+1)
            bin_edges = [dist.ppf(p,loc=loc,scale=scale,*arg) for p in cumulative_probability]
            observed_frequency, bin_edges = np.histogram(data_sample, bins=bin_edges, normed=False)
            c , p = st.chisquare(observed_frequency, expected_frequency, ddof=2)    
            dist_results.append([dist_name, round(c,4), round(p,4), param])
    return sorted(dist_results,key = lambda x:x[1])

def best_distribution(distr,rank,df):
    n = len(df)
    dist_name = distr[0]
    chisquare = distr[1]
    p_value = distr[2]
    param = distr[3]
    arg = param[:-2]
    loc = param[-2]
    scale = param[-1]
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(
        go.Histogram(x=df[0].values.tolist(), name="Observed Frequency"),
        secondary_y=False,
    )
    min_value, max_value = float(df.min()),float(df.max())
    steps = (max_value - min_value) / 20000
    xrange = np.arange(min_value,max_value, steps)
    dist = getattr(st, dist_name)
    code = f"scipy.stats.{dist_name}.rvs({','.join([str(p) for p in param])},size={n})"
    fig.add_trace(
        go.Scatter(x=xrange, 
                    y=dist(loc=loc,scale=scale,*arg).pdf(xrange),
                    name="Expected Probability"),
        secondary_y=True,
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Values")

    # Set y-axes titles
    fig.update_yaxes(title_text="Count", secondary_y=False)
    fig.update_yaxes(title_text="Probability", secondary_y=True)

    # Position the legends
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    ),
    margin={'t': 30}
    )
    card = styled_card(
                [
                    html.H3(f"{rank} - {DISTRIBUTIONS[dist_name]}"),
                    html.P(f"Chi Square Statistic = {chisquare} | P-value = {p_value}"),
                    html.P([dcc.Markdown(f'''Code to generate simulated data: ```{code}```''')]),
                    dcc.Graph(
                        id = dist_name,
                        figure = fig
                    )
                ]
            )
    return card

def qqplot(distr,df):
    n = len(df)
    dist_name = distr[0]
    chisquare = distr[1]
    p_value = distr[2]
    param = distr[3]
    arg = param[:-2]
    loc = param[-2]
    scale = param[-1]
    min_value = float(df.min())
    max_value = float(df.max())
    observed_quantile = (df.rank() - 0.5) / n
    dist = getattr(st,dist_name)
    theoretical_quantile = [dist.ppf(p,loc=loc,scale=scale,*arg) for p in observed_quantile[0].values.tolist()]
    slope, intercept, r_value, p_value, std_err = st.linregress(df[0].values.tolist(), theoretical_quantile)
    rsquare = r_value ** 2
    fig = go.Figure()
    fig.add_scatter(x=df[0].values.tolist(), y=theoretical_quantile, mode='markers')
    fig.add_shape(type="line",
        xref="x", yref="y",
        x0=min_value, y0=min_value, x1=max_value, y1=max_value,
        line=dict(
            color="Red",
            width=2
        ),
    )
    # Set x-axis title
    fig.update_xaxes(title_text="Observed Quantile")
    # Set y-axes titles
    fig.update_yaxes(title_text="Theoretical Quantile")
    fig.update_layout(showlegend=False)
    #fig.update_layout(shapes = [{'type': 'line', 'yref': 'paper', 'xref': 'paper', 'y0': 0, 'y1': 1, 'x0': 0, 'x1': 1}])
    card = styled_card(
                [
                    html.H3(f"QQ Plot of {DISTRIBUTIONS[dist_name]}"),
                    html.P(f"R Square = {rsquare}"),
                    dcc.Graph(
                        id = dist_name,
                        figure = fig
                    )
                ]
            )
    return card

def display_analysis(df, best_dist, name):
    if isinstance(df, str):
        if df == "error":
            return warning_message('There was an error processing this file.')                  
        elif df == "empty":    
            return warning_message("Upload your data to see the best distribution based on chi square test!")
    else:
        # Get Histogram of data
        simple_histogram = px.histogram(df)
        simple_histogram.update_layout(showlegend=False,margin={'t': 30})
        # Get Summary Statistic
        summary_statistics = df.describe()
        summary_statistics.loc['variance'] = df.var().tolist()
        summary_statistics.loc['skew'] = df.skew().tolist()
        summary_statistics.loc['kurtosis'] = df.kurtosis().tolist()
        summary_statistics.reset_index(inplace=True)
        summary_statistics.columns = ["Statistics","Value"]
        summary_statistics.Value = summary_statistics.Value.apply(significant_figure)

        return [
            dbc.Row(
                [
                    dbc.Col(
                        styled_card(
                            [
                                html.H3(f"Histogram for data in {name}"),
                                dcc.Graph(
                                    id = "Histogram",
                                    figure = simple_histogram
                                )
                            ]
                        ),width=9
                    ),
                    dbc.Col(
                        styled_card(
                            [
                                html.H3("Summary Statistics"),
                                dash_table.DataTable(
                                    data=summary_statistics.to_dict('records'),
                                    columns=[{'name': i, 'id': i} for i in summary_statistics.columns],
                                    style_cell={
                                        'padding': '5px',
                                        'fontSize': 15,
                                        'font-family':'sans-serif'
                                    },
                                    style_header={
                                        'backgroundColor': 'white',
                                        'fontWeight': 'bold',
                                        'text-align': 'center'
                                    },
                                    style_table={'overflowX': 'auto'},
                                )
                            ]
                        ),width=3
                    ),
                ]
            ),
            dbc.CardBody(
                [
                    dbc.Row(
                        dbc.Col(
                            children = [
                                best_distribution(best_dist[0],'1st',df),
                                qqplot(best_dist[0],df)
                            ]
                        )
                    ),
                    dbc.Row(
                        dbc.Col(
                            children = [
                                best_distribution(best_dist[1],'2nd',df),
                                qqplot(best_dist[1],df)
                            ]
                        )
                    ),
                    dbc.Row(
                        dbc.Col(
                            children = [
                                best_distribution(best_dist[2],'3rd',df),
                                qqplot(best_dist[2],df)
                            ]
                        )
                    )
                ]
            )
            
        ]

# UI UTILITY FUNCTIONS
def styled_card(elements):
    card_style = {
                "border-radius" : "5px",
                "background-color" : "#FFFFFF",
                "margin": "10px",
                "position": "relative",
                "box-shadow": "2px 2px 2px lightgrey"
                }
    return dbc.Card(
        dbc.CardBody(
            elements
        ),style=card_style
    )

def warning_message(msg):
    return html.Div(
        dbc.Row(
            dbc.Col(
                styled_card(
                    html.H5(msg)
                )
            )
        )
    )

navbar = dbc.NavbarSimple(
            children=[
                dbc.NavItem(dbc.NavLink("Home", href="https://tingwei.fans")),
                dbc.NavItem(dbc.NavLink("Portfolio", href="https://tingwei.fans/portfolio")),
                dbc.NavItem(dbc.NavLink("Blog", href="https://https://blog.tingwei.fans"))
            ],
            brand="Ting Wei"
        )
app.title = "Input Modelling"

# APP UI served here
app.layout = html.Div(
    children = [
        navbar,
        html.Center(
            children = [
                html.Br(),
                dbc.Row(html.H1("Input Modelling")),
                dbc.Row(html.H4("Chi-Square Goodness of Fit Test")),
            ]
        ),
        dbc.Row([
            dbc.Col(
                styled_card(
                        [
                            html.H3("What is Chi Square Test?"),
                            html.P(DESCRIPTION),
                            dbc.ListGroup(
                                [
                                    dbc.ListGroupItem("1. Identify candidate distribution that might fit the data well"),
                                    dbc.ListGroupItem("2. Take a sample of the data. (Chi-Square Test performs badly when sample size is too large)"),
                                    dbc.ListGroupItem("3. Calculate the maximum likelihood estimators for parameters for each candidate distributions"),
                                    dbc.ListGroupItem("4. Compute bins edge that results in equal probability of occurence in each bin based on each candidate distribution CDF and the corresponding expected frequency"),
                                    dbc.ListGroupItem("5. Compute observed frequency in class interval using the bins computed in previous step"),
                                    dbc.ListGroupItem(["6. Compute the Chi Square statistic:", html.Img(src=app.get_asset_url('chisquare.png'))]),
                                    dbc.ListGroupItem("7. Identify the best distributions by the lowest Chi Square statistic"),
                                    dbc.ListGroupItem("8. Verify the best distributions by looking at QQ Plot")
                            ]
                        ),
                    ]
                )
            )
        ]),
        dbc.Row(
                    dbc.Col(
                        [
                            styled_card(
                                [   
                                    html.H3("Input Area"),
                                    html.Br(),
                                    html.P("Upload your data here in csv/excel format"),
                                    html.P("Only a single column of numerical data without headers is allowed"),
                                    dcc.Upload(
                                        id='upload-data',
                                        children=html.Div([
                                            'Drag and Drop or ',
                                            html.A('Select Files')
                                        ],id='uploadContent'),
                                        style={
                                            'width': '100%',
                                            'height': '60px',
                                            'lineHeight': '60px',
                                            'borderWidth': '1px',
                                            'borderStyle': 'dashed',
                                            'borderRadius': '5px',
                                            'textAlign': 'center',
                                            },
                                    ),
                                    html.Br(),
                                    html.P("1. Candidate Distribution"),
                                    dcc.Dropdown(
                                        id="candidate_distribution",
                                        options=DISTRIBUTION_OPTIONS,
                                        value=['norm','gamma','expon','lognorm'],
                                        multi=True
                                    ),
                                    html.Br(),
                                    html.P("2. Sampling (percentage if between 0 and 1, sample size if larger than 1)"),
                                    dbc.Input(
                                        id="sample_params",
                                        type="number",
                                        min=0,
                                        placeholder="e.g. 0.2 or 200",
                                        step="0.0001",
                                        value=0.2
                                    ),
                                    html.Br(),
                                    html.P("3. Number of bins for Chi Square Test"),
                                    dbc.Input(
                                        id="no_of_bins",
                                        type="number",
                                        min=0,
                                        placeholder="e.g. 10",
                                        value=10
                                    ),
                                ]
                            )
                        ]
                    )
        ),
        html.Div(id='output-data-upload') 
    ]
)

@app.callback([Output('output-data-upload', 'children'),
               Output('uploadContent', 'children')],
              [Input('upload-data', 'contents'),
               Input('candidate_distribution', 'value'),
               Input('no_of_bins', 'value'),
               Input('sample_params', 'value'),
               State('upload-data', 'filename')]
)
def update_output(content, candidate_distribution, no_of_bins , sample_params, name):
    print(candidate_distribution, sample_params, no_of_bins)
    data_check = False
    sampling_check = False
    no_of_bins_check = False
    # load data
    if content == None:
        raise PreventUpdate
    
    df = parse_contents(content, name)
    if isinstance(df,str) == False:   
        filename = html.P(f"{name} - scroll down for results",style={'color': 'mediumslateblue', 'fontSize': 25})
        data_check=True
        n = len(df)
        if sample_params == None:
            return warning_message("Please enter a value for sampling proportion / size"), filename
        elif sample_params < 0:
            return warning_message("Invalid Sampling Parameter - Sampling cannot be negative"), filename
        elif sample_params >= 0 and sample_params <= 1:
            sample_size = int(sample_params * n)
            sampling_check = True
        elif sample_params > 1 and sample_params < n:
            sample_size = sample_params
            sampling_check = True
        else:
            return warning_message("Invalid Sampling Parameter - Sample size cannot be larger than number of observation"), filename
        
        if no_of_bins == None:
            return warning_message("Please enter a value for number of bins"), filename
        elif no_of_bins < 0:
            return warning_message("Number of bins cannot be negative!"), filename
        elif no_of_bins > len(df):
            return warning_message("Number of bins cannot be more than number of observations!"), filename
        else:
            no_of_bins_check = True

    top_dist = "None"
    if no_of_bins_check and sampling_check and data_check:
        top_dist = get_best_dist(df.values.tolist(),candidate_distribution,no_of_bins,sample_size)[:4]
    return display_analysis(df,top_dist, name), filename

if __name__ == '__main__':
    app.run_server(host="0.0.0.0", port=8050)