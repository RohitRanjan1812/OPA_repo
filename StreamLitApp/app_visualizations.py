import altair as alt
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

alt.data_transformers.disable_max_rows()

@st.cache_resource
def chart_quant(df, data_type, feature1, feature2):
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

    df = df.reset_index()
    features.insert(0, 'symbol')
    # # create a new DataFrame that stacks all columns of the original DataFrame
    # stacked_df = df[features].stack().reset_index()
    # stacked_df.columns = ['index', 'key', 'value']
    
    # # create text chart
    # text_chart = alt.Chart(stacked_df)\
    #         .mark_text()\
    #         .encode(
    #             alt.X(
    #                 "key",
    #                 type="nominal",
    #                 axis=alt.Axis(
    #                     orient="top",
    #                     labelAngle=0,
    #                     title=None,
    #                     ticks=False,
    #                     labelColor='#FFFFFF'
    #                 ),
    #                 scale=alt.Scale(align=0.5, padding=10),
    #                 sort=None,
    #             ),
    #             alt.Y("index", type="ordinal", axis=None),
    #             alt.Text("value", type="nominal"),
    #             color=alt.condition(brush, alt.value('#90EE90'), alt.value('#ddd'), legend=None) # Color text based on selection
    #         )\
    #         .properties(height=5000)
    # Define the scatter plot
    if data_type == 'quantitative':
        scatter = alt.Chart(df[features]).mark_circle().encode(
            x=feature1,
            y=feature2,
            color=alt.condition(brush, alt.value('#90EE90'), alt.value('#ddd')), # Color points based on selection
            tooltip='symbol'

        ).transform_filter(
            brush
        ).interactive()
    else:
        scatter = alt.Chart(df[features]).mark_circle().encode(
            x=feature1,
            y=feature2,
            color=alt.condition(brush, alt.value('#90EE90'), alt.value('#ddd')), # Color points based on selection
            size=alt.Size('count(symbol)', scale=alt.Scale(range=[5, 100])),
            tooltip= 'count()' #alt.Tooltip(field='symbol:N', title='Symbols', type='nominal', aggregate='values')  #'values(symbol)'

        ).transform_filter(
            brush
        ).interactive()
    # scatter_tooltip = alt.Chart(source.reset_index().groupby([feature1, feature2])['symbol'].apply(list).reset_index()).mark_circle().encode(
    #     x=feature1,
    #     y=feature2,
    #     color=alt.condition(brush, alt.value('#90EE90'), alt.value('#ddd')), # Color points based on selection
    #     size=alt.Size('count(symbol)', scale=alt.Scale(range=[5, 100])),
    #     tooltip=alt.Tooltip('symbol', delay=0)
    # ).transform_filter(
    #     brush
    # )
        
    # layer the two charts & repeat
    chart1 = alt.layer(
        background,
        highlight,
        data=source
    ).repeat(repeat=features[1:], columns=4) # type: ignore
    chart2 = scatter
    concat_chart = alt.hconcat(chart1, chart2, spacing=50).properties(
        autosize=alt.AutoSizeParams(
            type='fit',
            contains='padding')
    )   #.configure_axisX(orient='top') #.configure_view(height=5000, width=2000)
    st.vega_lite_chart(concat_chart.to_dict(),
    use_container_width=True) # type: ignore
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
        .mark_line(color='#90EE90') # type: ignore
        .encode(*encoding)
        .properties(width=chart_width, height=chart_height)
        .add_selection(brush)
    )

    # Create a chart for the selected region
    selected_region_chart = (
        alt.Chart(ts_df)
        .mark_line(color='#90EE90') # type: ignore
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
