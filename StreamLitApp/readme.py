# Import required libraries
import streamlit as st

def main():

    # Define title and subtitle
    st.markdown("Welcome to the <span style='font-weight: bold; color:#4BDCFF'>Portfolio Allocator tool</span>. \
                Please go through this 'readme' to effectively use this tool", unsafe_allow_html=True)

    with st.expander("Introduction", expanded=True):
        st.markdown('''
        The purpose of the application is to aid an Invester to select a subset of S&P 500, which minimizes risk by selecting diverse (mutually independent) stocks
        and further optimizes risk/profitability by adopting a smart weight allocation amongst the selected stocks. Further a time series prediction is performed
        for the effective portfolio and a return prediction is done over a time horizon.
        
        There are 2 different underlying datasets used in this application:
        1. Fundamental information: company information detailing both qualitative and quantitative aspects like sector, industry, EBIDTA, Margin etc
        2. Technical information: Time series data with the stock prices, volumn etc.
        <br>
        <br>
        The application is divided into three fundamental pillers:<p>
        
        1. EDA or Exploratory Data Analysis to explore the underlying Qualitative, Quantitative and time series data<br>
        2. Stock Selection via Clusturing analysis and ranking<br>
        3. Optimization of the allocation of the weights of individual ticker per alternative portfolio and prediction of return via TIme Series Models
        <br>
        <br>
        The approach taken for this application is to first apply various clusturing algorithms on the dataset with fundamental information and finally select the model best suited together with
        the hyperparameters. This allows us to use parameters relevant for Fundamental Analysis to perform diverse stock selection
        Moreover, we also use a Laplacian Score to rank important features which is used to rank and select stocks per cluster.
        
        Once we have defined our alternative portfolios forllowing stock selection we use multiple optimization techniques to derive weights for each ticker.
        
        To further aid the invester in the decision making, the time series data is used for predicting future return for each portfolio alternatives
        ''', unsafe_allow_html=True)

    with st.expander("EDA: Visualization"):
        st.write('''
        This tab helps the user to observe interactive charts, perform standalone cross filteration and study impact of individual features.
        The Qualitative and Quantitative tabs brings features from the Fundamental information dataset. In case the user would like to select multiple bars at the same time
        press shift and hold then click another bar on the chart for multi selection.
        The time series tab helps vizualize the evolution of stock metric over time.
        ''')

    with st.expander("Stock Selection"):
        st.write("""
        In this tab we present the user with a choice of 14 different portfolio choices with the stock composition and some fundamental information
        to base the selection choice
        """)

    with st.expander("Optimization and return analysis"):
        st.write("""
        Given the portfolio selection we display a pie chart with the estimated optimal weight distribution which has been derived by a
        Multi-Layer Perceptrone optimization technique applied to the historic time series data.
        Moreover a basic model information is displayed with the predicted and test time interval. 
        The user can choose the model and enter a time duration for prediction from the last available dataset. The final outcome is a % return
        """)

