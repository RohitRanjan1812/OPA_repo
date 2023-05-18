import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import streamlit as st
import altair as alt

df = pd.read_pickle(r'D:\Py_Prjs\OPA_repo\data\df_allInfo_clean.pkl')
def table(df):
    return (
        alt.Chart(df.reset_index())
        .mark_text()
        .transform_fold(df.columns.tolist())
        .encode(
            x=alt.X(
                "key", # type: ignore
                type="nominal", # type: ignore
                axis=alt.Axis(
                    # flip x labels upside down
                    orient="top", # type: ignore
                    # put x labels into horizontal direction
                    labelAngle=0, # type: ignore
                    title=None, # type: ignore
                    ticks=False # type: ignore
                ),
                scale=alt.Scale(padding=10), # type: ignore
                sort=None, # type: ignore
            ),
            y=alt.Y("index", type="ordinal", axis=None), # type: ignore
            text=alt.Text("value", type="nominal"), # type: ignore
        )
    )
def test_graph(df, feature, brush, filter):
    # brush = alt.selection_single(on='click', empty='none')

    # Create the histogram
    histogram = alt.Chart(df).mark_bar().encode(
        x=alt.X(f"{feature}", bin=alt.Bin(maxbins=30)), # type: ignore
        y=alt.Y('count()', title='Count'), # type: ignore
        color=alt.condition(brush, alt.value('lightgreen'), alt.value('lightgray')),
        tooltip=[alt.Tooltip(f'{feature}', bin=alt.Bin(maxbins=30), title=feature), 'count()'] # type: ignore
    ).properties(
        width=600,
        height=400
    ).add_selection(
        brush
    ).transform_filter(
        filter
    ).interactive()

    # Add the chart to your Streamlit app
    st.altair_chart(histogram, use_container_width=True)
    return histogram
