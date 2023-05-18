import streamlit as st
import preprocessing_data as ppd
import numpy as np
import pandas as pd
import app_visualizations as viz
import readme as rm
import methodology as mt
# import stockSelection as stl
df = pd.read_pickle(r'D:\Py_Prjs\OPA_repo\data\df_allInfo_clean.pkl')
df = ppd.fix_tickerInfo_dtypes(df)
ts_df=pd.read_pickle(r'D:\Py_Prjs\OPA_repo\data\50yr_timeSeries_data.pkl')
# Set page configurations
st.set_page_config(
    page_title="Portfolio Allocator Tool",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

def main():
    st.title('Portfolio Allocator Tool')
    st.write('The application helps an invester on selecting stocks out of S&P500 and determine optimal allocation and return predictions')
    readme, methodology, tab_visualization, tab_stockSelection, tab_returnAnalysis = st.tabs(["Readme", 'methodology', "EDA: Visualizations", "Stock Selection", "Optimization and Return Analysis"])
    with readme:
        rm.main()
    with methodology:
        st.write('test')
        mt.docu()
    with tab_visualization:
        # Create a container to hold the buttons
        button_container = st.container()
        with button_container:
            # Add some buttons to the container
            header, button1, button2, button3 = st.columns([10, 1, 1, 1])
            with header:
                st.subheader("Exploratory data Analysis: drill downs & visualizations")
            with button1:
                button1_clicked = st.button("Quantitative")
            with button2:
                button2_clicked = st.button("Qualitative")
            with button3:
                button3_clicked = st.button("Time Series")
            # Check if each button was clicked
            if button1_clicked:
                st.experimental_set_query_params(button="quantitative")
            if button2_clicked:
                st.experimental_set_query_params(button="qualitative")
            if button3_clicked:
                st.experimental_set_query_params(button="time_series")
            # Get the state of the button from the query parameters
            button = st.experimental_get_query_params().get("button", None)
            if button is not None:
                button = button[0]

            if button == "quantitative":
                viz.chart_quant(df, 'quantitative')
            elif button == "qualitative":
                viz.chart_quant(df, 'nominal')
            elif button == "time_series":
                tickers, metrics = ppd.get_tickers_metrics_ts(ts_df)
                selected_ticker = st.selectbox('Select a Ticker', tickers)
                selected_metric = st.selectbox('Select a Metric', metrics)
                viz.time_series_charts(ppd.get_timeSeries(ts_df, selected_ticker, selected_metric))



    with tab_stockSelection:
        st.write('test')
    with tab_returnAnalysis:
        st.title("The selected tab is for Optimization and Return Analysis")

if __name__ == '__main__':
    main()
