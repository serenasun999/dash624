#!/usr/bin/env python3
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


def load_data(url):
    """
    Load data from a shared google drive csv
    :param url: the shared url string
    :returns: a pandas dataframe
    """
    # file_id = url.split("/")[-2]
    a4_data = pd.read_csv(url)
    # a4_data = pd.read_csv("./dataset/2023-02-08-DATA624-Assignment4-Data.csv")

    # dwn_url = "https://drive.google.com/uc?id=" + file_id
    # df = pd.read_csv(dwn_url)
    return a4_data


url = "https://raw.githubusercontent.com/serenasun999/dash624/master/dataset/2023-02-08-DATA624-Assignment4-Data.csv"
a4_data = load_data(url)

def kmeans(dataset):
    kmeans = KMeans(
        n_clusters = 4, # Number of clusters to find
        init = "k-means++", # How to place the initial cluster centroids,
        max_iter= 100, # Maximum number of iterations for the algorithm to run
        tol=0.0001, # Roughly how much the centroids need to change between iterations to keep going
    ).fit(
        dataset
    )

    data_kmeans =pd.concat([
        dataset,
        pd.DataFrame(
            kmeans.labels_,
            columns=['cluster']
        ).astype('category')
    ], axis=1)

    fig1 = px.scatter(
        data_kmeans, 
        x='0', 
        y='1', 
        color='cluster',
        color_continuous_scale='Set1',
        width=800,
        height=600,
        title='Clustering with KMeans'
    )
    return fig1
    # return color_map,colors,data_kmeans

def agglomerativeClustering(dataset):
    # In this regard, single linkage is the worst strategy, and Ward gives the most regular sizes. 
    # However, the affinity (or distance used in clustering) cannot be varied with Ward, thus for 
    # non Euclidean metrics, average linkage is a good alternative. Single linkage, while not robust 
    # to noisy data, can be computed very efficiently and can therefore be useful to provide hierarchical 
    # clustering of larger datasets. Single linkage can also perform well on non-globular data.

    clusters_agg = AgglomerativeClustering(
        n_clusters = 4, # Number of clusters to find
        linkage="average",
    ).fit(
        dataset
    )

    data_agg =pd.concat([
        dataset,
        pd.DataFrame(
            clusters_agg.labels_,
            columns=['cluster']
        ).astype('category')
    ], axis=1)

    fig2 = px.scatter(
        data_agg, 
        x='0', 
        y='1', 
        color='cluster',
        color_continuous_scale='Set1',
        width=800,
        height=600,
        title='Clustering with AgglomerativeClustering'
    )
    return fig2
# plot scatter plot with colors based on 'cluster' column

def dbscan(dataset):
    clusters = DBSCAN(
    # eps= 0.5, # Max distance between two points to assign to same cluster 
    eps= 2.5, # Max distance between two points to assign to same cluster 
    # eps= 2, # Max distance between two points to assign to same cluster 
    ).fit(
        dataset
    )

    data_dbscan =pd.concat([
        dataset,
        pd.DataFrame(
            clusters.labels_,
            columns=['cluster']
        ).astype('category')
    ], axis=1)

    fig3 = px.scatter(
        data_dbscan, 
        x='0', 
        y='1', 
        color='cluster',
        color_continuous_scale='Set1',
        width=800,
        height=600,
        title='Clustering with DBSCAN'
    )
    return fig3
    # return color_map,colors,data_dbscan


def tsne(dataset):
    ts = TSNE(
        perplexity=100, # Roughly the "size" of the clusters to look for (original paper
                    # recommends in the 5-50 range, but in general should be less than
                    # then number of points in your dataset
        learning_rate="auto",
        n_iter=500,
        init='pca',
    ).fit_transform(dataset)
    fig4 = px.scatter(
            x=ts[:, 0],
            y=ts[:, 1],
            color=dataset['label'],
            width=600,
            height=600,
            title='t-SNE Visualization'
        )
    return fig4


labels = a4_data.iloc[:,14]
a4_data_df = a4_data.iloc[:,:13]
a4_data_df

a4_data_df = pd.concat([
    pd.DataFrame(a4_data_df),
    pd.DataFrame(a4_data.iloc[:,14]),
],axis=1)
a4_data_df.columns = [str(x) for x in a4_data_df.columns]
last_col_name = a4_data_df.columns[-1]
# create a dictionary of column names to rename
new_col_names = {last_col_name: 'label'}
# rename the columns using the rename() method
a4_data_df = a4_data_df.rename(columns=new_col_names)

fig1 = kmeans(a4_data_df)
fig2 = agglomerativeClustering(a4_data_df)
fig3 = dbscan(a4_data_df)
fig4 = tsne(a4_data_df)

app.layout = html.Div(
    [
        html.H1("Lecture 3 -- Distributions"),
        """
        Let's explore the sepsis data!
        Not the best styling so hopefully you can improve it.
        The code shows some parameters you can manipulate, but there are lots more to try!
        """,
        dcc.Graph(
            figure=fig1,
            style={
                "width": "80%",
                "height": "70vh",
            },
            id="OurFirstFigure",
        ),
        dcc.Graph(
            figure=fig2,
            style={
                "width": "100%",
                "height": "50vh",
            },
            id="OurSecondFigure",

        ),
        dcc.Graph(
            figure=fig3,
            style={
                "width": "40vh",
                "height": "40vh",
            },
            id="OurThirdFigure",

        ),
        dcc.Graph(
            figure=fig4,
            style={
                "width": "40vh",
                "height": "40vh",
            },
            id="OurFourthFigure",

        ),
    ]
)

if __name__ == "__main__":
    app.run_server(debug=False)
    # kmeans(a4_data)