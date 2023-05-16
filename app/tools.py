from funcs.get_segmentation_datasets_info import get_segmentation_datasets_info
from funcs.segment_dataset import segment_dataset
from funcs.pdai import pdai
from langchain.agents import Tool

get_segmentation_datasets_info_tool = Tool(
    name="get_segmentation_datasets_info",
    func=get_segmentation_datasets_info,
    description="""
    Get information about the segmentation datasets in json format.
    This function takes no inputs and returns a jsonified dictionary with the following keys:

    ARGS
    ________________
    NONE. 

    RETURNS
    ________________
    A JSON with the following keys:
    - path: The path to the dataset
    - description: A description of the dataset
    """
)

segment_dataset_tool = Tool(
    name="segment_dataset",
    func=segment_dataset,
    description="""
    Segment a dataset given its filepath and the number of samples to take from the dataset. 
    Returns a jsonified pd dataframe with the segmented data.

    ARGS
    ________________
    A JSON with the following keys:
    - n: The number of samples to take from the dataset
    - path_to_dataset: The path to the dataset
    """
)

pdai_tool = Tool(
    name="pdai",
    func=pdai,
    description="""
    Manipulate and/or plot a dataframe given a natural language prompt

    ARGS
    ________________
    A JSON with the following keys:
    - prompt: The prompt to use
    - path: The filepath to the csv file with data to manipulate
    MUST BE STRICTLY IN JSON FORMAT

    RETURNS
    ________________
    A string with the status of the operation
    """
)