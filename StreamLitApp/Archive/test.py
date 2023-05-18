import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import streamlit as st
import altair as alt

def test_graph(df, feature, brush): #, filter):
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
    ).interactive()
    #     .transform_filter(
    #     filter
    # )\
    # Add the chart to your Streamlit app
    st.altair_chart(histogram, use_container_width=True)
    return histogram
def test_graph_filter(df, feature, brush, filter):
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
def main():
    df = pd.read_pickle(r'D:\Py_Prjs\OPA_repo\data\df_allInfo_clean.pkl')
    key = 1
    brush = dict()
    feature = st.selectbox('select a stock feature', df.columns, key=key)
    # filter = alt.selection_single(empty='none', on='click', fields=[feature], init={feature: None}) # type: ignore
    brush[key] = alt.selection_single(on='click', empty='none', encodings=['x'])
    chart = test_graph(df, feature, brush[key]) #, filter)
    filter = brush[key]
    key = key + 1
    feature = st.selectbox('select a stock feature', df.columns, key=key)
    brush[key] = alt.selection_single(on='click', empty='none', encodings=['x']) # type: ignore
    chart = test_graph_filter(df, feature, brush[key], filter)
    filter = brush[key]
    key = key + 1

if __name__ == '__main__':
    main()