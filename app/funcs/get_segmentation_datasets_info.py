import os
import json

def get_segmentation_datasets_info(dummy : str)-> str:
    """
    Get the datasets from the data folder.

    Returns:
        string: A jsonified dictionary with the path to the datasets and their descriptions.
    """
    datasets = {}

    current_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.dirname(current_dir)
    data_dir = os.path.join(parent_dir, 'datasets/segmentation')

    for dataset_dir in os.listdir(data_dir):
        path_to_dataset_folder = os.path.join(data_dir, dataset_dir)
        path_to_description = os.path.join(path_to_dataset_folder, 'description.txt')
        file_name = os.path.basename(path_to_dataset_folder)
        path_to_dataset = os.path.join(path_to_dataset_folder, f'{file_name}.csv')

        if os.path.isdir(path_to_dataset_folder) == False:
            raise Exception(f'{path_to_dataset_folder} is not a directory.')
        if os.path.isfile(path_to_description) == False:
            raise Exception(f'{path_to_description} is not a file.')
        if os.path.isfile(path_to_dataset) == False:
            raise Exception(f'{path_to_dataset} is not a file.')

        description = ""
        try:
            with open(path_to_description, 'r') as f:
                description = f.read()
        except Exception as e:
            raise Exception(f'Error reading {path_to_description}: {e}')
        
        datasets[dataset_dir] = {
            'path': path_to_dataset,
            'description': description
        }
    return json.dumps(datasets)