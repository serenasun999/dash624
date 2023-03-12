#!/usr/bin/env python3
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
import geopandas as gp
import plotly.graph_objects as go
import plotly.express as px
import geopandas as gpd
import shapely.geometry
import numpy as np


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


def load_data():
    """
    Load data from a shared google drive csv
    :param url: the shared url string
    :returns: a pandas dataframe
    """

    a6_data_speed = pd.read_csv("https://raw.githubusercontent.com/serenasun999/dash624/master/dataset/Speed_Limits.csv")
    a6_data_camera = pd.read_csv("https://raw.githubusercontent.com/serenasun999/dash624/master/dataset/Traffic_Cameras.csv")
    a6_data_school = pd.read_csv("https://raw.githubusercontent.com/serenasun999/dash624/master/dataset/School_Locations.csv")

    return a6_data_speed, a6_data_camera,a6_data_school

# url = "https://raw.githubusercontent.com/serenasun999/dash624/master/dataset/2023-02-08-DATA624-Assignment4-Data.csv"
a6_data_speed, a6_data_camera,a6_data_school = load_data()

def school_gpd(school):
    school['the_geom'] = gp.GeoSeries.from_wkt(school['the_geom']) # resign it to mention it is a geometry file
    school = school.set_geometry('the_geom') 
    return school

def speed_gpd(speed):
    speed['multiline'] = gp.GeoSeries.from_wkt(speed['multiline']) # resign it to mention it is a geometry file
    speed = speed.set_geometry('multiline')
    return speed

def camera_gpd(camera):
    camera['Point'] = gp.GeoSeries.from_wkt(camera['Point']) # resign it to mention it is a geometry file
    camera = camera.set_geometry('Point')
    return camera

def convertMultiLine(speed_data):
    lats = []
    lons = []
    names = []

    for feature, name in zip(speed_data.geometry, speed_data.SPEED):
        if isinstance(feature, shapely.geometry.linestring.LineString):
            linestrings = [feature]
        elif isinstance(feature, shapely.geometry.multilinestring.MultiLineString):
            linestrings = feature.geoms
        else:
            continue
        for linestring in linestrings:
            x, y = linestring.xy
            lats = np.append(lats, y)
            lons = np.append(lons, x)
            names = np.append(names, [name]*len(y))
            lats = np.append(lats, None)
            lons = np.append(lons, None)
            names = np.append(names, None)


    return lats, lons, names

def drawMap(lats, lons, names, school_data, camera_data):
    fig = px.line_mapbox(lat=lats, lon=lons, hover_name=names,
                     mapbox_style="carto-positron", zoom=1)

    fig.add_trace(go.Scattermapbox(
                        lat=school_data.geometry.y,
                        lon=school_data.geometry.x,
                        mode='markers',
                        text=school_data.GRADES,
                        hoverinfo='text',
                        marker=dict(
                            color='plum',
                        ),
                        name='Schools'
                        ))

    fig.add_trace(go.Scattermapbox(
                        lat=camera_data.geometry.y,
                        lon=camera_data.geometry.x,
                        mode='markers',
                        text=camera_data.Quadrant,
                        hoverinfo='text',
                        name='Traffic Cameras'
                        ))

    fig.update_layout(mapbox_zoom=10,
                    mapbox_center_lat=51.0486, mapbox_center_lon=-114.0708,
                    margin={"r":0,"t":0,"l":0,"b":0})

    return fig

school_data = school_gpd(a6_data_school)
speed_data = speed_gpd(a6_data_speed)
camera_data = camera_gpd(a6_data_camera)
lats, lons, names = convertMultiLine(speed_data)
fig = drawMap(lats,lons,names,school_data,camera_data)

app.layout = html.Div(
    [
        html.H1("Assignment 6"),
        """
        Let's explore the sepsis data!
        Not the best styling so hopefully you can improve it.
        The code shows some parameters you can manipulate, but there are lots more to try!
        """,
        dcc.Graph(
            figure=school_data.plot(),
            style={
                "width": "80%",
                "height": "70vh",
            },
            id="OurFirstFigure_score",
        ),
        dcc.Graph(
            figure=speed_data.plot(),
            style={
                "width": "80%",
                "height": "70vh",
            },
            id="OurFirstFigure",
        ),
        dcc.Graph(
            figure=camera_data.plot(),
            style={
                "width": "80%",
                "height": "70vh",
            },
            id="OurSecondFigure_score",

        ),
        dcc.Graph(
            figure=fig,
            style={
                "width": "80%",
                "height": "70vh",
            },
            id="OurSecondFigure",
        )
    ]
)

if __name__ == "__main__":
    app.run_server(debug=False)