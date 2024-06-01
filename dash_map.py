from sklearn.linear_model import LogisticRegression
from mplsoccer import VerticalPitch
import matplotlib.pyplot as plt
import numpy as np

from dash import Dash, html, dcc, callback, Output, Input, State
import plotly.graph_objs as go
import plotly.express as px
import plotly.graph_objects as go
import plotly_football_pitch as pfp

# RUN THIS - Custom make_pitch fct

"""Create a plotly figure of a football pitch."""
from typing import Optional

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from plotly_football_pitch.pitch_background import PitchBackground
from plotly_football_pitch.pitch_dimensions import PitchDimensions, PitchOrientation
from plotly_football_pitch.pitch_plot import make_ellipse_arc_svg_path



def make_pitch_figure(
    dimensions: PitchDimensions,
    marking_colour: str = "black",
    marking_width: int = 4,
    pitch_background: Optional[PitchBackground] = None,
    figure_width_pixels: int = 800,
    figure_height_pixels: int = 600,
    orientation: PitchOrientation = PitchOrientation.HORIZONTAL,
) -> go.Figure:
    """Create a plotly figure of a football pitch with markings.

    Some markings which appear in both team's halves e.g. penalty box arc, are
    defined in terms of the attacking team and the defending team. For a
    horizontally oriented pitch the attacking team's half is the left hand one,
    while for a vertically oriented one their half is the bottom one.

    Args:
        dimensions (PitchDimensions): Dimensions of the pitch to plot.
        marking_colour (str): Colour of the pitch markings, default "black".
        marking_width (int): Width of the pitch markings, default 4.
        pitch_background (Optional[PitchBackground]): Strategy for plotting a
            background colour to the pitch. The default of None results in a
            transparent background.
        figure_width_pixels (int): Width of the figure, default 800. This
            corresponds to the long axis of the pitch (pitch length).
        figure_height_pixels (int): Height of the figure, default 600. This
            corresponds to the short axis of the pitch (pitch width).
        orientation (PitchOrientation): Orientation of the pitch, defaults to
            a horizontal pitch.

    Returns:
        plotly.graph_objects.Figure
    """
    pitch_marking_style = {
        "mode": "lines",
        "line": {"color": marking_colour, "width": marking_width},
        "hoverinfo": "skip",
        "showlegend": False,
    }
    spot_style = {
        "mode": "markers",
        "line": {"color": marking_colour},
        "hoverinfo": "skip",
        "showlegend": False,
    }

    touchline = {
        "x": [0, dimensions.pitch_length_metres, dimensions.pitch_length_metres, 0, 0],
        "y": [0, 0, dimensions.pitch_width_metres, dimensions.pitch_width_metres, 0],
    }

    halfway_line = {
        "x": [dimensions.pitch_mid_length_metres, dimensions.pitch_mid_length_metres],
        "y": [0, dimensions.pitch_width_metres],
    }

    # attacking team's half is left for horizontal pitches (bottom for
    # vertical after rotation)
    penalty_boxes = {
        "attacking_team": {
            "x": [
                0,
                dimensions.penalty_box_length_metres,
                dimensions.penalty_box_length_metres,
                0,
            ],
            "y": [
                dimensions.penalty_box_width_min_metres,
                dimensions.penalty_box_width_min_metres,
                dimensions.penalty_box_width_max_metres,
                dimensions.penalty_box_width_max_metres,
            ],
        },
        "defending_team": {
            "x": [
                dimensions.pitch_length_metres,
                dimensions.pitch_length_metres - dimensions.penalty_box_length_metres,
                dimensions.pitch_length_metres - dimensions.penalty_box_length_metres,
                dimensions.pitch_length_metres,
            ],
            "y": [
                dimensions.penalty_box_width_min_metres,
                dimensions.penalty_box_width_min_metres,
                dimensions.penalty_box_width_max_metres,
                dimensions.penalty_box_width_max_metres,
            ],
        },
    }

    six_yard_boxes = {
        "attacking_team": {
            "x": [
                0,
                dimensions.six_yard_box_length_metres,
                dimensions.six_yard_box_length_metres,
                0,
            ],
            "y": [
                dimensions.six_yard_box_width_min_metres,
                dimensions.six_yard_box_width_min_metres,
                dimensions.six_yard_box_width_max_metres,
                dimensions.six_yard_box_width_max_metres,
            ],
        },
        "defending_team": {
            "x": [
                dimensions.pitch_length_metres,
                dimensions.pitch_length_metres - dimensions.six_yard_box_length_metres,
                dimensions.pitch_length_metres - dimensions.six_yard_box_length_metres,
                dimensions.pitch_length_metres,
            ],
            "y": [
                dimensions.six_yard_box_width_min_metres,
                dimensions.six_yard_box_width_min_metres,
                dimensions.six_yard_box_width_max_metres,
                dimensions.six_yard_box_width_max_metres,
            ],
        },
    }

    penalty_spots = {
        "attacking_team": {
            "x": [dimensions.penalty_spot_length_metres],
            "y": [dimensions.pitch_mid_width_metres],
        },
        "defending_team": {
            "x": [
                dimensions.pitch_length_metres - dimensions.penalty_spot_length_metres
            ],
            "y": [dimensions.pitch_mid_width_metres],
        },
    }

    centre_spot = {
        "x": [dimensions.pitch_mid_length_metres],
        "y": [dimensions.pitch_mid_width_metres],
    }

    pitch_marking_coordinates_with_style = [
        (touchline, pitch_marking_style),
        (halfway_line, pitch_marking_style),
        (penalty_boxes["attacking_team"], pitch_marking_style),
        (penalty_boxes["defending_team"], pitch_marking_style),
        (six_yard_boxes["attacking_team"], pitch_marking_style),
        (six_yard_boxes["defending_team"], pitch_marking_style),
        (penalty_spots["attacking_team"], spot_style),
        (penalty_spots["defending_team"], spot_style),
        (centre_spot, spot_style),
    ]

    pitch_markings = [
        go.Scatter(
            **orientation.switch_axes_if_required(marking_coordinates),
            **style,
        )
        for marking_coordinates, style in pitch_marking_coordinates_with_style
    ]

    
    
    
    fig = make_subplots()
    for markings in pitch_markings:
        fig.add_trace(markings)

    centre_circle = {
        "x0": dimensions.pitch_mid_length_metres
        + dimensions.centre_circle_radius_metres,
        "y0": dimensions.pitch_mid_width_metres
        + dimensions.centre_circle_radius_metres,
        "x1": dimensions.pitch_mid_length_metres
        - dimensions.centre_circle_radius_metres,
        "y1": dimensions.pitch_mid_width_metres
        - dimensions.centre_circle_radius_metres,
    }
    fig.add_shape(
        type="circle",
        xref="x",
        yref="y",
        line=pitch_marking_style["line"],
        name=None,
        **orientation.switch_axes_if_required(
            centre_circle, keys_to_switch=[("x0", "y0"), ("x1", "y1")]
        ),
    )

    penalty_box_arcs = {
        "attacking_team": {
            "x_centre": dimensions.penalty_spot_length_metres,
            "y_centre": dimensions.pitch_mid_width_metres,
        },
        "defending_team": {
            "x_centre": dimensions.pitch_length_metres
            - dimensions.penalty_spot_length_metres,
            "y_centre": dimensions.pitch_mid_width_metres,
        },
    }
    start_angle = {
        "attacking_team": {
            PitchOrientation.HORIZONTAL: -np.pi / 3,
            PitchOrientation.VERTICAL: np.pi / 6,
        },
        "defending_team": {
            PitchOrientation.HORIZONTAL: 2 * np.pi / 3,
            PitchOrientation.VERTICAL: -5 * np.pi / 6,
        },
    }
    end_angle = {
        "attacking_team": {
            PitchOrientation.HORIZONTAL: np.pi / 3,
            PitchOrientation.VERTICAL: 5 * np.pi / 6,
        },
        "defending_team": {
            PitchOrientation.HORIZONTAL: 4 * np.pi / 3,
            PitchOrientation.VERTICAL: -np.pi / 6,
        },
    }

    path = make_ellipse_arc_svg_path(
        **orientation.switch_axes_if_required(
            penalty_box_arcs["attacking_team"],
            keys_to_switch=[("x_centre", "y_centre")],
        ),
        start_angle=start_angle["attacking_team"][orientation],
        end_angle=end_angle["attacking_team"][orientation],
        a=dimensions.penalty_spot_length_metres,
        b=dimensions.penalty_spot_length_metres,
        num_points=60,
    )
    fig.add_shape(
        type="path",
        path=path,
        line=pitch_marking_style["line"],
        name=None,
    )
    path = make_ellipse_arc_svg_path(
        **orientation.switch_axes_if_required(
            penalty_box_arcs["defending_team"],
            keys_to_switch=[("x_centre", "y_centre")],
        ),
        start_angle=start_angle["defending_team"][orientation],
        end_angle=end_angle["defending_team"][orientation],
        a=dimensions.penalty_spot_length_metres,
        b=dimensions.penalty_spot_length_metres,
        num_points=60,
    )
    fig.add_shape(
        type="path",
        path=path,
        line=pitch_marking_style["line"],
        name=None,
    )

    if pitch_background is not None:
        fig = pitch_background.add_background(fig, dimensions, orientation)

    axes_ranges = {
        "xaxis_range": [0, 0.5*dimensions.pitch_length_metres],
        "yaxis_range": [0, dimensions.pitch_width_metres],
    }

    
    # HAD TO REMOVE UPDATING SIZE OF FIGURE MANUALLY - IT CAUSED HUGE PROBLEMS FOR CENTERING,
    # I RESCALE IT LATER WHEN CREATING THE dcc.Graph !
    fig.update_layout(
#         height=figure_height_pixels,
#         width=figure_width_pixels,
        **orientation.switch_axes_if_required(
            axes_ranges, keys_to_switch=[("xaxis_range", "yaxis_range")]
        ),
    )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig





# From: https://soccermatics.readthedocs.io/en/latest/gallery/lesson2/plot_xGModelFit.html

b = [-0.5103,  -0.6338, 0.2798]
model_variables = ['Angle','Distance']
model=''
for v in model_variables[:-1]:
    model = model  + v + ' + '
model = model + model_variables[-1]

#return xG value for more general model
def calculate_xG(sh, foot = 0):
   bsum=b[0]
   for i,v in enumerate(model_variables):
       bsum=bsum+b[i+1]*sh[v] + foot * 0.2
   xG = 1/(1+np.exp(bsum))
   return xG



def make_fig(left = 0):
    #Create a 2D map of xG
    pgoal_2d = np.zeros((68,68))
    for x in range(68):
        for y in range(68):
            sh=dict()
            a = np.arctan(7.32 *x /(x**2 + abs(y-68/2)**2 - (7.32/2)**2))
            if a<0:
                a = np.pi + a
            sh['Angle'] = a
            sh['Distance'] = np.sqrt(x**2 + abs(y-68/2)**2)
            pgoal_2d[x,y] =  calculate_xG(sh, left)

    dimensions = pfp.PitchDimensions()
    fig = make_pitch_figure(
        dimensions,
        figure_height_pixels=800,
        figure_width_pixels=600,
        orientation=pfp.PitchOrientation.VERTICAL,
        pitch_background= pfp.SingleColourBackground("#81B622")
    )

    heatmap_trace = go.Heatmap(z=pgoal_2d, colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']], hoverinfo='none', showscale=False)
    fig.add_trace(heatmap_trace)

    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        modebar=dict(orientation='h'),
        dragmode = False
    )

    # fig.update_layout(annotations=[annotation])
    # Attach the callback function to the click event
    # heatmap_trace.on_click(update_annotation)
    return fig







# Initialize the app
app = Dash(__name__)
server = app.server

# App layout
app.layout = html.Div([
    html.Div(children='TEST', style={'textAlign': 'center'}),
    html.Hr(),
    dcc.RadioItems(options=[
        {'label': 'Left-foot', 'value': 'Left-foot'},
        {'label': 'Right-foot', 'value': 'Right-foot'}
    ], value='Right-foot', id='controls-and-radio-item', style={'textAlign': 'center', 'marginBottom': '20px'}),
    dcc.Store(id='stored-click-data', data=None),  # Hidden store for click data
    html.Div(style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'justifyContent': 'center', 'width': '100%'}, children=[
        html.Div([
            dcc.Markdown("""
                **Goal Probability**
            """, style={'fontSize': '24px', 'border': '2px solid black', 'padding': '10px', 'backgroundColor': '#f9f9f9', 'textAlign': 'center'}),
            html.Pre(id='click-data', style={'fontSize': '24px', 'color': 'green', 'border': '2px solid black', 'padding': '10px', 'backgroundColor': '#f9f9f9', 'textAlign': 'center'}),
        ], style={'marginBottom': '20px', 'textAlign': 'center'}),
        html.Div(style={'display': 'flex', 'justifyContent': 'center', 'width': '100%'}, children=[
            dcc.Graph(figure={}, id='controls-and-graph', style={'width': '600px', 'height': '400px'}, config={'displayModeBar': False})
        ])
    ])
])

# Add controls to build the interaction
@app.callback(
    Output('controls-and-graph', 'figure'),
    [Input('controls-and-radio-item', 'value'),
     Input('stored-click-data', 'data')]
)
def update_graph(col_chosen, storedData):
    if col_chosen == 'Left-foot':
        fig = make_fig(1)
    elif col_chosen == 'Right-foot':
        fig = make_fig(0)
        
    # Add a red dot if storedData is not None
    if storedData is not None and storedData.get('clickData') is not None:
        clickData = storedData['clickData']
        x = clickData['points'][0]['x']
        y = clickData['points'][0]['y']
        z = fig.data[-1].z[y][x]  # Accessing the last heatmap data
        fig.add_trace(go.Scatter(
            x=[x],
            y=[y],
            mode='markers',
            marker=dict(color='red', size=25),
            hoverinfo='none',
            showlegend=False  # Prevent the red dot from appearing in the legend
        ))
        fig.add_trace(go.Scatter(
            x=[x],
            y=[y-0.25],
            mode='text',
            text='P',
            textposition='middle center',
            textfont=dict(color='white', size=20),
            showlegend=False  # Prevent the text from appearing in the legend
        ))

    return fig

@app.callback(
    Output('stored-click-data', 'data'),
    [Input('controls-and-graph', 'clickData'),
     Input('controls-and-radio-item', 'value')],
    State('stored-click-data', 'data')
)
def store_click_data(clickData, radioValue, storedData):
    if clickData is not None:
        return {'clickData': clickData, 'radioValue': radioValue}
    # Retain previous stored data if no new click
    return storedData

@app.callback(
    Output('click-data', 'children'),
    [Input('stored-click-data', 'data'),
     Input('controls-and-radio-item', 'value')]
)
def display_click_data(storedData, radioValue):
    if storedData and storedData.get('clickData'):
        clickData = storedData['clickData']
        x = clickData['points'][0]['x']
        y = clickData['points'][0]['y']
        if radioValue == 'Left-foot':
            fig = make_fig(1)
        else:
            fig = make_fig(0)
        z = fig.data[-1].z[y][x]
        return round(z, 4)
    return None

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
