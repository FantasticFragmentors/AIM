import pandas as pd
import gower
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import streamlit as st
import plotly.express as px

def get_distance_matrix(df):
    """
    Calculates the distance matrix for a given dataset.

    Parameters
    ----------
    df: pandas.DataFrame
        The data to be clustered.

    Returns
    -------
    distance_matrix: array-like, shape (n_samples, n_features)
        The distance matrix for the data.
    """
    for column in tqdm(df.columns):
        if df[column].dtype == 'int64':
            df[column] = df[column].astype('float64')
    distance_matrix = gower.gower_matrix(df)
    return distance_matrix

def preprocess(df):
    """
    Removes null values and columns with too many unique values

    Parameters
    ----------
    df: pandas.DataFrame
        The data to be preprocessed.

    Returns
    -------
    df: pandas.DataFrame
        The preprocessed data.

    df_columns_dropped: pandas.DataFrame
        The columns dropped from the data.
    """
    working_df = df.copy(deep=True)
    working_df.dropna(inplace=True)
    df_columns_dropped = pd.DataFrame()
    for column in working_df.columns:
        if working_df[column].nunique() > 0.9*working_df.shape[0]:
            df_columns_dropped[column] = working_df[column]
            working_df.drop(columns=[column], inplace=True)
    return working_df, df_columns_dropped

def tune_model(distance_matrix):
    """
    Tunes the clustering model and returns the best hyperparameters.

    Parameters
    ----------
    distance_matrix: array-like, shape (n_samples, n_features)
        The data to be clustered.

    Returns
    -------
    n_clusters: int
        The ideal number of clusters for the data.
    """
    n_clusters = 2
    best_score = -1
    for i in tqdm(range(2, 10)):
        clustering_model = AgglomerativeClustering(n_clusters=i, metric='precomputed', linkage='complete')
        labels = clustering_model.fit_predict(distance_matrix)
        score = silhouette_score(distance_matrix, labels)
        if score > best_score:
            best_score = score
            n_clusters = i
    print("Best score: {best_score}, n_clusters: {n_clusters}".format(best_score=best_score, n_clusters=n_clusters))
    return n_clusters

def get_labels(distance_matrix):
    """
    Fits a model to the data and returns the cluster labels.
    https://towardsdatascience.com/clustering-on-numerical-and-categorical-features-6e0ebcf1cbad
    https://www.geeksforgeeks.org/ml-hierarchical-clustering-agglomerative-and-divisive-clustering/

    Parameters
    ----------
    distance_matrix: array-like, shape (n_samples, n_features)
        The data to be clustered.

    Returns
    -------
    labels: array-like, shape (n_samples,)
        Cluster labels for each point in the dataset given to fit_predict.
    """
    n_clusters = tune_model(distance_matrix)

    clustering_model = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='complete')

    labels = clustering_model.fit(distance_matrix).labels_

    score = silhouette_score(distance_matrix, labels)
    print("Score: {best_score}, n_clusters: {n_clusters}".format(best_score=score, n_clusters=n_clusters))

    return labels

def generate_plot_df(distance_matrix, labels):
    pca = PCA(n_components=2)
    pca.fit(distance_matrix)
    X_PCA = pca.transform(distance_matrix)

    x, y = X_PCA[:, 0], X_PCA[:, 1]

    plot_df = pd.DataFrame({'x': x, 'y':y, 'label':labels}) 

    return plot_df

    
def plot_clusters(plot_df):
    """
    Plots the clusters in a 2D space.

    Parameters
    ----------
    distance_matrix: array-like, shape (n_samples, n_features)
        The data to be clustered.
    labels: array-like, shape (n_samples,)
        Cluster labels for each point in the dataset given to fit_predict.
    """

    fig = px.scatter(plot_df, x="x", y="y", color="label", title="Clustered Data")

    st.plotly_chart(fig)