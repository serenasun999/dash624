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
from plotly.subplots import make_subplots



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

a6_data_speed, a6_data_camera,a6_data_school = load_data()

def school_gpd(school):
    school['the_geom'] = gp.GeoSeries.from_wkt(school['the_geom']) # resign it to mention it is a geometry file
    school["Quadrant"] = school["ADDRESS"].str[-2:]
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

def barfig(school_data, camera_data):
    school = school_data.groupby(["Quadrant","GRADES"]).count().reset_index()
    camera = camera_data.groupby("Quadrant").count().reset_index()
    fig = make_subplots(rows=1, cols=2)

    fig.add_trace(go.Bar(x=school["Quadrant"], y=school["NAME"], text=school["GRADES"],
                        base='overlay', name="Schools"),1,1)

    fig.add_trace(go.Bar(x=camera["Quadrant"], y=camera["Point"], name="Cameras"),1,2)

    fig.update_xaxes(title_text="Quadrant", row=1, col=1)
    fig.update_yaxes(title_text="Number of Schools", row=1, col=1)
    fig.update_xaxes(title_text="Quadrant", row=1, col=2)
    fig.update_yaxes(title_text="Number of Cameras", row=1, col=2)

    # update the layout to display the subplots side by side
    fig.update_layout(title='Distribution of Schools and Cameras in Calgary')

    return fig



school_data = school_gpd(a6_data_school)
speed_data = speed_gpd(a6_data_speed)
camera_data = camera_gpd(a6_data_camera)
lats, lons, names = convertMultiLine(speed_data)
fig = drawMap(lats,lons,names,school_data,camera_data)
fig_1 = barfig(school_data,camera_data)

app.layout = html.Div(
    [
        html.H1("Assignment 6"),
        """
        This assignment will look at three datasets (School location, Traffic camera, and speed limit map) from the city of Calgary. We are trying to proportion of camera installed around school area, and how speed limit designed around school area, and anywhere cameras are installed.

        - School Locations: https://data.calgary.ca/Services-and-Amenities/School-Locations/fd9t-tdn2 \n

        - Traffic Camera: https://data.calgary.ca/Transportation-Transit/Traffic-Cameras/k7p9-kppz/explore \n

        - Speed Limit Map: https://data.calgary.ca/Health-and-Safety/Speed-Limits-Map/rbfp-3tic \n
        """,
        html.H2("Distribution of Schools and Cameras in Calgary"),
        dcc.Graph(
            figure=fig_1,
            style={
                "width": "80%",
                "height": "70vh",
            },
            id="OurSecondFigure_score",
        ),
        """
        Based on the quadrant, there are over 60 schools located in NW, and in SE and SW over 50 schools are located in these two area. Relatively fewer shcools are located in NE area. Most Elementary schools are distributed equally across Calgary but in NW has greater number of Elementary and Senior High schools.

        Based on the quadrant, most cameras are installed in SE since downtown area might be included in and usually downtown has the most cameras installed and has the most traffic in a city. In other three areas (NE,NW,SW) has similar number of cameras installed.

        Initially, the question I would like to discover is that whether there is an asscociation between number of schools and number of cameras in Calgary. After comparing two bargraphs, the result shows the number of schools is not associated with the number of cameras installed in Calgary.
        The next step is to discover relationship between distribution of cameras installed, speed limitation and schools in Calgary.
        """,
        html.H2("Distribution of Schools, Cameras, and Speed Limit in Calgary on Map"),
        dcc.Graph(
            figure=fig,
            style={
                "width": "80%",
                "height": "70vh",
            },
            id="OurSecondFigure",
        ),
        """
        As a result, we could not conclude that there is an association between number of schools, number of cameras and speed limit. The area has more schools do not indicate number of installed camera in that area is higher than others. The most cameras are installed on roads in which has speed limit greater than 50km/h,
        it might because people are more likely to speeding on highways. Generally, schools are loacted close to major roads and cameras are installed in major traffic areas. From the map we could discover, except downtown area, the number of camera installed is equally disributed in Calgary.

        This map can help to decide where to install new cameras based on speed limit, school zones and incident reports. Also base on incident reports, this map can help to design speed limit across Calgary.Furthermore, it can help to resign education resources with community population.
        """
    ]
)

if __name__ == "__main__":
    app.run_server(debug=False)