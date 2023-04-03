import streamlit as st
import eda_viz_module as eda
# Set page configurations
st.set_page_config(
    page_title="Portfolio Allocator Tool",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

def main():
    st.title('Portfolio Allocator Tool')
    st.write('The application helps an invester on selecting stocks out of S&P500\
              and determine optimal allocation and return predictions')
    tab_visualization, tab_stockSelection, tab_returnAnalysis = st.tabs(["EDA: Visualizations", "Stock Selection", "Optimization and Return Analysis"])
    with tab_visualization:
        st.subheader("The selected tab is for EDA: Visualizations")
        eda.test_graph(eda.df)
    with tab_stockSelection:
        st.title("The selected tab is for Stock Selection")
    with tab_returnAnalysis:
        st.title("The selected tab is for Optimization and Return Analysis")

if __name__ == '__main__':
    main()
