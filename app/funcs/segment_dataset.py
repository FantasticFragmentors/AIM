import streamlit as st
import os
import json
import pandas as pd
from .pipelines import segmentation

def segment_dataset(args : str)-> str:
    """
    Segment a dataset given its filepath

    Args:
        args (str): A jsonified dictionary with the following keys:
            n: The number of samples to take from the dataset
            path_to_dataset: The path to the dataset
    Returns:
        string: A jsonified dictionary with the segmented data.
    """

    args = json.loads(args)
    n = args['n']
    path_to_dataset = args['path_to_dataset']

    #Keep it very small for now to save money w/ open AI
    #The data is pre segemnted so we can just sample from it
    n = 1000

    current_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.dirname(current_dir)

    data_file_name = os.path.basename(path_to_dataset)
    segmented_data_dir = os.path.join(parent_dir, 'datasets/segmented')
    segmented_data_path = os.path.join(segmented_data_dir, f'{data_file_name}')
    segmented_data_plot_path = os.path.join(segmented_data_dir, f'{data_file_name.split(".")[0]}_plot.csv')

    # To save compute, store the segmented data in a file
    # If a saved file exists, return it instead of rerunning the model
    if os.path.isfile(segmented_data_path) and os.path.isfile(segmented_data_plot_path):
        segmented_data = pd.read_csv(segmented_data_path)
        segmented_data_plot = pd.read_csv(segmented_data_plot_path)
        if segmented_data.shape[0] == n:
            return finish(segmented_data, segmented_data_plot)
        if segmented_data.shape[0] >= n:
            segmented_data = segmented_data.sample(n=n, random_state=42)
            return finish(segmented_data, segmented_data_plot)
    
    raw_data = pd.read_csv(path_to_dataset)

    sampled_data = raw_data.sample(n=n, random_state=42)

    preprocessed_data, columns_dropped = segmentation.preprocess(sampled_data)

    distance_matrix = segmentation.get_distance_matrix(preprocessed_data)
    labels = segmentation.get_labels(distance_matrix)

    clustered_data = preprocessed_data.copy(deep = True)
    clustered_data['cluster'] = labels

    final_clustered_data = pd.concat([clustered_data, columns_dropped], axis=1)
    final_clustered_data.to_csv(segmented_data_path, index=False)

    segmented_data_plot = segmentation.generate_plot_df(distance_matrix=distance_matrix, labels=labels)
    segmented_data_plot.to_csv(segmented_data_plot_path, index=False)
    
    return finish(final_clustered_data, segmented_data_plot)

def finish(df, plot_df):
    segmentation.plot_clusters(plot_df)
    df = df.apply(lambda x: pd.to_numeric(x, errors='coerce'))
    descriptive = df.groupby('cluster').agg(['mean', 'std'])
    # Flatten the multi-level index
    descriptive.columns = ['_'.join(col).strip() for col in descriptive.columns.values]
    st.dataframe(descriptive)
    return json.dumps(descriptive.to_dict())

