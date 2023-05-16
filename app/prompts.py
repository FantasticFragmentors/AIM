from langchain.prompts import PromptTemplate

describe_segments_template = PromptTemplate(
    input_variables=["dataset"],
    template="""

    BACKGROUND
    ________________

    You are a data scientist tasked with segmenting customers for a marketing campaign. \
    You have a CSV file with customer data that has already been clustered using an \
    unsupervised machine learning model, and the cluster that each customer belongs to \
    is indicated in the CSV file. You must interpret what each of these clusters mean \
    so that the marketing team can use this information to create targeted campaigns. \
    
    HINTS
    ________________

    1. The membership of each customer in a cluster is denoted by the 'cluster' column,
    and each cluster is a positive integer.

    2. Customers in different clusters are differentiated by the ML model because of significant \
    differences in the values of the remaining columns. Identifying these differences can be helpful \
    in understanding what each cluster means.

    3. All the clusters that exist are numbered in the CSV file. For example, if the 'cluster' column \
    only contains integer values 0 and 1, there are only two clusters: 0 and 1.\

    4. Make sure to describe every cluster in the CSV file. For example, if the 'cluster' column \
    contains integer values 0, 1, and 2, there are three clusters: 0, 1, and 2.\
    
    5. Your answer should be bulleted and in MARKDOWN format
     
    TASK
    ________________
    
    Identify and describe the clusters in the given CSV dataset in simple, non-technical language \
    so that the marketing team can understand. Write the description of each cluster in a bulleted \
    list. 

    DATA
    ________________
    {dataset}
    """)

auto_segment_template = PromptTemplate(
    input_variables=["query"],
    template="""

    BACKGROUND
    ________________
    You are a data scientist tasked with segmenting customers for a marketing campaign. \
    You have CSV files that contain customer data. You must pick only one dataset that \
    contains data that is most relevant to the advertising campaign in question. You will then use \
    and unsupervised machine learning model to cluster the customers and interpret what each of these clusters \
    mean so that the marketing team can use this information to create targeted campaigns. \
    
    CLUSTERING HINTS
    ________________

    1. The membership of each customer in a cluster is denoted by the 'cluster' column,
    and each cluster is a positive integer.

    2. Customers in different clusters are differentiated by the ML model because of significant \
    differences in the values of the remaining columns. Identifying these differences can be helpful \
    in understanding what each cluster means.

    3. All the clusters that exist are numbered in the CSV file. For example, if the 'cluster' column \
    only contains integer values 0 and 1, there are only two clusters: 0 and 1.\

    4. Make sure to describe every cluster in the CSV file. For example, if the 'cluster' column \
    contains integer values 0, 1, and 2, there are three clusters: 0, 1, and 2.\
    
    5. Your answer should be bulleted and in MARKDOWN format
     
    TASK
    ________________
    
    You are given a question by a marketer about their advertisement campaign. Find out what dataset \
    is most relevant to the question and segment the dataset. Then, interpret what each of the clusters \
    mean so that the marketing team can use this information to create targeted campaigns. 
    Your answer should be bulleted and in MARKDOWN format.\
    
    EXAMPLE
    ________________
    - Cluster 1: This cluster contains customers who are young and have a high income. They are likely to \
    be interested in the product.

    - Cluster 2: This cluster contains customers who are old and have a low income. They are unlikely to \
    be interested in the product.

    WARNINGS
    ________________
    - There is no tool to describe or analyze the clusters for you. You must do this yourself. The output \
    should be like the example and in plain english. Once you reach this step exit the langchain. \
    
    - You have an answer length limit of 257 characters. This means that your answer should be short. \
    DO NOT EXCEED THE LIMIT.

    MARKETER'S QUESTION
    ________________
    {query}
    """)

pdai_template = PromptTemplate(
    input_variables=["query"],
    template="""

    BACKGROUND
    ________________
    You are a data scientist with a lot of datasets. You want to analyze these datasets. \
    You have a tool that can list the paths to all these datasets.\
    You have a tool (pdai) that can analyze and/or plot a dataset given a natural language prompt. \
    Your answer should be bulleted and in MARKDOWN format.
    
    TASK
    ________________
    - Use the describe_segments tool to find out the filepaths of the datasets \
    - Pick the right dataset filepath and use the pdai tool to analyze and/or plot the dataset. \
    
    PROMPT
    ________________
    {query}
    """
)