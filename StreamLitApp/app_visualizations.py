import altair as alt
import pandas as pd
import numpy as np
import streamlit as st

@st.cache_resource
def chart_quant(df, data_type):
    source = df

    brush = alt.selection(type='multi', encodings=['x']) # type: ignore

    # Define the base chart, with the common parts of the
    # background and highlights
    base = alt.Chart().mark_bar().encode(
        x=alt.X(alt.repeat('repeat'), type=data_type if data_type == 'quantitative' else 'nominal', bin=alt.Bin(maxbins=10) if data_type == 'quantitative' else None), # type: ignore
        y=alt.Y('count()', title=''), # type: ignore
        color=alt.value('#90EE90')
    ).properties(
        width=160,
        height=130
    )

    # gray background with selection
    background = base.encode(
        color=alt.value('#ddd')
    ).add_selection(brush)

    # blue highlights on the transformed data
    highlight = base.transform_filter(brush)
    if data_type == 'quantitative':
        features = df.select_dtypes(include=[np.number]).columns.tolist()
    elif data_type == 'nominal':
        features = df.select_dtypes(include='object').columns.tolist()
    # layer the two charts & repeat
    st.vega_lite_chart(alt.layer(
        background,
        highlight,
        data=source
    ).repeat(repeat=features, columns=4).to_dict()) # type: ignore
    return brush

@st.cache_resource
def time_series_charts(ts_df):
    # Combine x and y encoding for both charts
    encoding = alt.X("date:T"), alt.Y("value:Q") # type: ignore

    # Create a selection brush for the x-axis (time)
    brush = alt.selection_interval(encodings=["x"])

    # Set the chart width and height
    chart_width, chart_height = 1500, 500

    # Create the line chart for the time series data with brush selection
    line_chart_with_brush = (
        alt.Chart(ts_df)
        .mark_line()
        .encode(*encoding)
        .properties(width=chart_width, height=chart_height)
        .add_selection(brush)
    )

    # Create a chart for the selected region
    selected_region_chart = (
        alt.Chart(ts_df)
        .mark_line()
        .transform_filter(brush)
        .encode(
            x=alt.X(
                "date:T", scale=alt.Scale(domain=brush.ref())  # type: ignore
            ),
            y=alt.Y(
                "value:Q", # type: ignore
                scale=alt.Scale(domainMax=float(ts_df["value"].max()), domainMin=float(ts_df["value"].min())),  # type: ignore
            ),
        )
        .properties(width=chart_width, height=chart_height)

    )

    # Display the interactive time series chart
    interactive_chart = line_chart_with_brush & selected_region_chart
    st.altair_chart(interactive_chart)



