#!/usr/bin/env python3

from dash import Dash, html
import dash_bootstrap_components as dbc

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div(
    [
        html.H1("Layout Prototype"),
        """
        This is just dummy text for now
        There will be something cool going on later!!!
        """,
    ]
)

if __name__ == "__main__":
    app.run_server(debug=False)
