import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pacmap
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

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
    for i in range(2, 10):
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
    
def plot_clusters(distance_matrix, labels):
    """
    Plots the clusters in a 2D space.

    Parameters
    ----------
    distance_matrix: array-like, shape (n_samples, n_features)
        The data to be clustered.
    labels: array-like, shape (n_samples,)
        Cluster labels for each point in the dataset given to fit_predict.
    """

    pca = PCA(n_components=2)
    pca.fit(distance_matrix)
    X_PCA = pca.transform(distance_matrix)
    X_PCA.shape

    x, y = X_PCA[:, 0], X_PCA[:, 1]

    # colors = {0: 'red',
    #         1: 'blue',
    #         2: 'green', 
    #         3: 'yellow'}

    # names = {0: 'who make all type of purchases', 
    #         1: 'more people with due payments', 
    #         2: 'who purchases mostly in installments', 
    #         3: 'who take more cash in advance'} 
    
    df = pd.DataFrame({'x': x, 'y':y, 'label':labels}) 
    groups = df.groupby('label')

    fig, ax = plt.subplots(figsize=(12, 8)) 

    for name, group in groups:
        ax.plot(group.x, group.y, 
                marker='o', 
                linestyle='',
                ms=5,
                # color=colors[name],
                # label=names[name], 
                label=name,
                mec='none')
        ax.set_aspect('auto')
        ax.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
        ax.tick_params(axis= 'y',which='both',left='off',top='off',labelleft='off')
        
    ax.legend()
    ax.set_title("Clustered Data")
    plt.show()