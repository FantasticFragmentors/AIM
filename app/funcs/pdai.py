import io
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI

llm = OpenAI()

def pdai(args : str)->str:
    """
    Uses pandas ai to manipulate or visualize dataframes given a prompt

    Parameters
    ----------
    args: str
        A json string with the prompt and the path to the data to be manipulated or visualized.

    Returns
    -------
    str
        A string that denotes completion
    """

    args = json.loads(args)
    prompt = args['prompt']
    path = args['path']

    df = pd.read_csv(path)

    pandas_ai = PandasAI(llm, conversational=False)
    result = pandas_ai.run(df, prompt=prompt)

    output_string = output_string = """
            Pandas AI has completed its task.

            OUTPUT
            ------
            {result}
            """.format(result=result)

    if type(result) == pd.DataFrame:
        st.dataframe(data=result)
    elif type(result) == np.ndarray:
        for row in result:
            for ax in row:
                fig = ax.get_figure()
                st.pyplot(fig)
    elif type(result) == str:
        pass
    else:
        fig = result.get_figure()
        st.pyplot(fig)
    return output_string
