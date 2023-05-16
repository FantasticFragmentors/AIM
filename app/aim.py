# streamlit run aim.py
import os
import pandas as pd
import streamlit as st
import agents
import prompts
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory


datasets, pdaichat, aimchat = st.tabs (["Datasets", "PDAI Chat", "AIM Chat"])

with aimchat:
    st.title("AIM Chat")
    prompt = st.text_input('How can I help you today?', key='aimchat')

    if prompt:
        full_prompt = prompts.auto_segment_template.format(query=prompt)
        segments = agents.aim_agent.run([full_prompt])
        st.markdown(segments)

with pdaichat:
    st.title("PDAI Chat")
    prompt = st.text_input('How can I help you today?', key='pdaichat')

    if prompt:
        full_prompt = prompts.pdai_template.format(query=prompt)
        response = agents.pdai_agent.run([full_prompt])
        st.markdown(response)

with datasets:
    st.title("Datasets")
    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(current_dir, 'datasets/segmentation')
    for dataset_dir in os.listdir(data_dir):
        path_to_dataset_folder = os.path.join(data_dir, dataset_dir)
        path_to_description = os.path.join(path_to_dataset_folder, 'description.txt')
        file_name = os.path.basename(path_to_dataset_folder)
        path_to_dataset = os.path.join(path_to_dataset_folder, f'{file_name}.csv')

        description = ""
        try:
            with open(path_to_description, 'r') as f:
                description = f.read()
        except Exception as e:
            raise Exception(f'Error reading {path_to_description}: {e}')
        
        dataset = pd.read_csv(path_to_dataset)

        st.subheader(dataset_dir)
        st.dataframe(data=dataset.head(3))
        show_description = st.checkbox('Show Description', key=dataset_dir)
        if show_description:
            st.markdown(description)
        


    



