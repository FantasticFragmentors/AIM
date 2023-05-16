import tools
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
from prompts import describe_segments_template, auto_segment_template

llm = OpenAI()

aim_tools = [
    tools.get_segmentation_datasets_info_tool,
    tools.segment_dataset_tool
]
aim_agent = initialize_agent(aim_tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

pdai_tools = [
    tools.get_segmentation_datasets_info_tool,
    tools.pdai_tool
]
pdai_agent = initialize_agent(pdai_tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)