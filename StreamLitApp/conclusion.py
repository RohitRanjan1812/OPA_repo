import streamlit as st

def docu():
    st.markdown('**Conclusion:**')
    st.write('1. The current work applies varies Machine Learning and Deep Learning techniques to create an Optimized Portfolio of stocks.')
    st.write('2. The clustering algorithms are used to impart an initial natural diversification of stocks just on the basis of Fundamental information available.')
    st.write('3. We created alternative portfolio based on clusturing and ranking based stock selction.')
    st.write('4. Optimized weight allocation per alternative portfolio based upon technical time series\
        information maximizes return while accepting the inherent volatility in the stock selection.')
    st.write('5. Time series models are used to provide an insight into the future outlook per selected alternative portfolio.')
    st.write('\n')
    st.markdown('**Scope for the future:**')
    st.write('1. The current work focuses only on S&P 500 data (to enable less computation time) and can be extended to a broader collection of stocks.')
    st.write('2. A ML-Ops based automated pipeline could also be built which updates the underlying data daily and also trains the models based on new information.')
    st.write('3. We could enable a live training and prediction based upon user based selection of stocks rather than the 14 alternative portfolios offered currently.')
    st.write('4. Implementation of short-selling via -ve weights to allow for a even better risk-return management')
    st.write('5. Incorporation of predicted inflation impact based upon macroeconomic data to allow for a realistic judgement of predicted future return.')
    st.write("6. Using Deep Learning methods on Time Series to perform better.")