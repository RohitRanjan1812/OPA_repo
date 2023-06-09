import streamlit as st
from PIL import Image


def _load_img():
    coll = dict()
    coll['hcp_dend'] = Image.open(r'D:\Py_Prjs\OPA_repo\data\images\hca_dendogram.png')
    coll['knn_opt'] = Image.open(r'D:\Py_Prjs\OPA_repo\data\images\knn_optimalCluster.png')
    coll['knn_bestfeatures'] = Image.open(r'D:\Py_Prjs\OPA_repo\data\images\knn_onSelectedFeatures.png')
    coll['Cluster_dist'] = Image.open(r'D:\Py_Prjs\OPA_repo\data\images\clustering\num_stocks_per_cluster_1.jpg')
    coll['cluster_evol_1'] = Image.open(r'D:\Py_Prjs\OPA_repo\data\images\clustering\cluster_evolution.jpg')
    coll['cluster_evol_2'] = Image.open(r'D:\Py_Prjs\OPA_repo\data\images\clustering\cluster_evolution_2.jpg')
    return coll
coll = _load_img()
@st.cache_data
def docu():
    with st.expander('Financial terms and terminologies used'):
        st.markdown('''**Portfolio**: A collection of stocks (in this work) on which an invester has invested.''')
        st.markdown('**Return**: The profit or loss an invester makes after a time period over an investment. It can also be represented in % change')            
        st.markdown('''**Volatility**: Prices of stocks (or there combination as in the case of Portfolio) changes over time and if this variability is high, we call the stock or the portfolio volatile
                    ''')
        st.markdown('**Ticker**: All stocks are listed in the market with a symbol. eg. AAPL, GOOG etc, these are tickers')
        st.markdown('**Stock evolution**: The change in prices of a stock and can be visualized in a time series chart.')
        st.markdown('**Fundamental information**: The published information regarding a stock based on balanced sheet performance and general news triggers.')
        st.markdown('**Technical information**: The price and volume movements and other reactive aspects of the stock, that could be studied as a pattern.')
    with st.expander("Clustering Analysis"):
        st.write('''
                This tab consists of technical details of results from the clustering analysis
                and supports the user to make informed decision on how to select one out of the 14 alternative portfolios
                ''')

        st.write('''
                We first look at the Hierarchical  Clustering Analysis, and after choosing a distance threshold of 1.75 based on best judgement
                we obtain a k value of 28. The dendogram is shown as:
                ''')

        st.image(coll['hcp_dend'])
        st.write('\n\n')
        st.write('''
                To trust and refine this number from HCA we need validation from k-means approach. A two-step approach is employed here -\n
                1. A KNN clustering analysis is done considering all the numerical features of the attribute dataaset.
                2. We incorporate an unsupervised feature selection using the laplacian score and only use the 21 important featues selected to perform the KNN analysis\n\n
                ''')
        img1, img2 = st.columns(2)
        with img1:
            st.image(coll['knn_opt'])
        with img2:
            st.image(coll['knn_bestfeatures'])
        st.write('\n\n')
        st.write('''
                Considering the inflection point at k = 26 from both the steps also indicate 
                that the information retained by the important features adequately represent the full attributes dataset. Moreover, this also validates the HCA results.
                Hence, k = 26 is selected for alternative portfolio stock selction.\n\n
                ''')
    with st.expander('Alternative Portfolios - Cluster size refinement'):
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(coll['Cluster_dist'])
        with col2:
            st.write('''
                    Now, looking at the distribution of the number of stocks per cluster:
                    A great discrepancy exists in the number of stocks contained within each cluster.
                    To create a balanced and diversified selections we employ a mean based reduction algorithm as follows:\n\n
                    1. Calculates the mean number of stocks over all clusters.\n
                    2. Categorises the clusters into groups with either more (above_mean) or less (below_mean) stocks than the overall mean.\n
                    3. Reduces the number of stocks within the above_mean cohort proportionally over all clusters within the above_mean cohort.\n
                    4. Keeps the number of stocks within the below_mean cohort unchanged.\n
                    5. Repeat above steps.\n\n
                    The next image shows the evolution of the number of stocks per cluster''')
        st.write('\n\n')
        col3, col4 = st.columns([3, 1])
        with col3:
            st.image(coll['cluster_evol_1'])
        with col4:
            st.image(coll['cluster_evol_2'])
    with st.expander('Alternative Portfolios - Stock Selection'):
        st.markdown('''Having clustered our stocks into 26 clusters (with initial alternative portfolio cluster_0), and determining how many stocks are required for each of these clusters (represented by alternative portfolio cluster_x), we also developed a ranking technique to select the number of stocks in alternative portfolio cluster_x out of the stocks in alternative portfolio cluster_0. For example, the alternative portfolio cluster_5 (refer to the image above with number of stocks within each cluster for cluster_x) with cluster labelled 8 we need 19 stocks out of the initial 62 in the cluster_0.
                    \n\nFor this we calculated a rank of importance for each stock in relation to all other stocks.
                    To calculate the rankings we used the 21 features and their corresponding lap scores (which signifies a columns' importance in relation to the other columns) calculated during feature selection.
                    \n\nThe calculation worked as follows:
                    \n1. For each feature column each stock received a ranking based on their column's value in relation to the other stocks' column values.
                    \n2. Using the lap scores, the rank for columns with lower importance received additional weighting penalty.
                    \n3. All the column rankings were summed together over each stock to create an importance rank for the stock.
                    \n4. The stocks with overall minimal score were ranked higher (eg. lowest overall score for a stock → rank 1)

                    \nWith the rank of importance calculated for each stock in relation to all other stocks we now had the means to select only the most highly ranked stocks in our shrunken down clusters. The process of selecting the most highly ranked stocks in each cluster for use in the alternative portfolio was iterated for all alternative portfolios / cluster_x.

                    \nThe result is 14 alternative portfolios, each containing 26 clusters, with each cluster containing only the most highly ranked stocks for the number of stocks dictated by our algorithm. Our next step was to apply weights to the stocks within each portfolios.''')
    with st.expander('Alternative Portfolios - Optimal Weight Allocation'):
        st.markdown('''
                We next proceeded to create 14 final alternative portfolios, each representing a clustering from the alternative portfolios cluster_0 to cluster_13 defined earlier. With each containing only the most highly ranked stocks.

                \nTo represent a portfolio we needed to determine the individual weights for the individual stocks.
                In simple terms the weight for a stock within a portfolio is the investment amount made in this stock as a percentage of the total investment made in all the stocks in the portfolio.

                \nThree methods were explored:
                Mean variance optimization (MVO)
                Modern portfolio theory with an assumption on risk free rate (MPT)
                Multilayer perceptron regressor (MLP)

                \nAs we already used the fundamental attributes dataset for clustering, we focused on using the time series data to perform the above mentioned methods to determine the weights.

                \n**Method 1: Mean variance optimization (MVO)**:
                The mean-variance optimization is a technique to minimise portfolio risk while achieving a target return. We first retrieved the historical stock price data and calculated daily returns, expected returns, and the covariance matrix. Using cvxpy,, a convex optimization library, we then optimised the asset allocation (weights) by minimising the portfolio variance, which represents the risk. The optimization is subject to constraints, such as achieving the target return and setting a minimum weight for each stock. The output is the optimal weights for the given portfolio.

                \n**Method 2: Modern portfolio theory with an assumption on risk free rate (MPT)**:
                This method constructs a portfolio using Modern Portfolio Theory (MPT) by incorporating a risk-free rate. It first loads the stocks and their historical price data, then calculates the daily returns, expected returns, and the covariance matrix. The objective function is defined to maximise the Sharpe ratio, which accounts for the risk-free rate*. The optimization problem is solved using the SciPy library, which finds the optimal weights subject to the constraint that the sum of the weights equals 1. The output is the optimal weights for the given portfolio according to MPT.

                \n*risk-free rate: historically investments in stocks are risky however it yields a higher return in the marker. Risk free rate refers to the interest rate an invester will get typically in a bond market where we have a low however a fixed guaranteed return (effectively meaning a minimal market risk)

                \nThe optimization problem is similar to MVO however the objective function changes to sharpe ratio which also accounts for the risk free rate i.e. the return a risk free investor gets elsewhere (particularly bond market).


                \n**Method 3: Multilayer perceptron regressor (MLP)**:
                The underlying code begins by loading stock tickers and historical price data, then calculates the daily returns. The returns are normalised using a MinMaxScaler and split into training and testing sets. An MLP model with two hidden layers is defined and trained on the training data. The trained MLP model is then used to predict the optimal weights for the testing data. The predicted weights for the last day are normalised, and the output displays the optimal weights for the given portfolio according to the MLP method.


                \nWe next needed to choose which optimiser to use. And from the outset there were several reasons why the MLP seemed best in class:

                \nNon-linearity: MLPs, as a type of artificial neural network, can model complex non-linear relationships between variables. Traditional methods like MVO assume a linear relationship between expected returns and risks. The ability of MLPs to capture non-linear relationships might improve the accuracy of predicted returns and subsequently the portfolio weights.

                \nAdaptability: MLPs can learn from historical data and adapt to changes in the market. By continuously updating the model with new data, MLPs can potentially provide more accurate and up-to-date portfolio weight recommendations compared to static optimization techniques.

                \nFeature engineering: MLPs can automatically learn relevant features from raw data, such as interactions between different stocks, without the need for manual feature engineering. This can save time and effort and may result in better predictions.

                \nFlexibility: MLPs offer flexibility in terms of architecture, activation functions, and optimization algorithms, which allows investors to experiment and fine-tune the model based on their specific requirements.

                \nAll 3 techniques were investigated, and we observed that the MVO and MPT based techniques yield either very low weights for the majority of the stocks and equal weights for the rest of the collection. Considering the weight distribution, non-linearity of cross stock interaction and historic performances MLP was selected as the go-to method for our use case.

                \nOur finalised 14 alternative portfolios were thus created using the MLP method, which were used as input into our time series model to predict future performance.

                ''')
    with st.expander('Time Series Analysis'):
        st.markdown('''
                \n**Modelling approach**:
                \n\nHaving created our alternative portfolios in terms of stock selection we now proceeded to create their time series equivalents.

                \n\nWe decided to limit the time period to dates between the years 2010 and 2019.
                And did so for the following reasons:
                \n1. It is a recent period and corresponds to how our attributes and time series datasets were chosen (using current stocks)
                \n2. There are no major financial crises within this period
                \n3. The US/China trade war did however cause a temporary drop of approx. 8.7% in August 2019 which introduces the impact of short term real world volatility

                \n\nUsing these 10 years we would train our models as follows:
                \nThe first 9 years of our time series data, 2010 to 2018, would be used for training.
                The next year, 2019, would be used as a test dataset for model development; as well as serve as a point of reference to compare the alternative portfolio's predicted performance against the actual market performance.

                \n\nNext we would develop our models as follows:
                \n1. Develop various different models.
                \n2. Train the models using the above mentioned approach
                \n3. Each model would be pre-trained on each alternative portfolio (as described above).''')

        st.markdown('''
                    \n\n**Portfolio Preparation**:
                    \n\nHaving decided our modelling approach we now had to prepare the time series data.

                    \nAs a start we noticed it only contained observations for business days.
                    Weekends and public holidays were missing.

                    \nTo negate the impact that this might have on modelling we created a version of the time series data that contained all the days, including weekends and public holidays. We then populated these newly added dates using linear interpolation.

                    \nNext we checked for nan values in the data and found that of the 503 stocks the overwhelming majority had very little nan values.


                    \n\nOur first decision upon seeing these results was to drop the 23 stocks that had more than 50% nan values for each of the 13 alternative portfolios.
                    This decision was based on the following reasons:
                    \n1. These stocks added insufficient value for modelling purposes.
                    \n2. Keeping these stocks and replacing their nan values with static values such as the stock median would result in the models training on static price points for the majority of training which would adversely impact our models' abilities to make accurate predictions.
                    \n3. Doing so would also ignore inflation, as the average stock prices during 2010 were in general far less than during 2019. 

                    \n\nNext we checked that each cluster within each alternative portfolio, following the above process, contained at least a single stock.
                    There was luckily only a single cluster in a single alternative portfolio that no longer had any stocks.
                    For this instance we simply added the stock in the same cluster from the nearest alternative portfolio to the empty cluster.

                    \n\nWe had thus finalised our alternative portfolios, along with their clusters and included stocks.
                    Next we had to apply weightings to each stock within its alternative portfolio and thus finalise the composition of each alternative portfolio.

                    \n\nTo calculate these weightings we used the MLP regressor developed earlier in the clustering piece. And checked the weighting do indeed sum to 1.
                    Next we created a new dataframe for each alternative portfolio whereby the stock weightings were applied.

                    \nFor each new data frame the values were also scaled according to the assumed investment start date.of 1 January 2019 and a fixed amount of USD 1000. This meant that for each alternative portfolio one need simply, without any adjustment, at each point in time during 2019, sum the values of each stock and that value would represent how an investment of $1000 on 1 January 2019 in that alternative portfolio would have developed.
                ''')

        st.markdown('''\n\n**Modelling**
                    \n\nFirst our data was split into training and testing sets by using the method explained above. The test period is needed in order to validate the appropriateness of the prediction models and also to make comparisons among them.
        \n\nThree time series analysis methods were implemented and investigated in detail:
        \n1. classical SARIMA model
        \n2. Linear regression methods
        \n3. Prophet forecasting

        \nFor each model different resampling methods were also evaluated (monthly or weekly) in order to improve the speed of the analysis. Predictions were evaluated against the test period, i.e. plot predictions were compared to test data and the root mean square error (RMSE) calculated. Where the RMSE allows us to compare the efficiency of all the methods for different resampling frequencies.
        \n\nProphet method stood out with the lowest RMSE (38 and 42 for weekly and montly resampling respectively). The second best was SARIMA which was a bit conservative in estimation.
        Both these models and the resampling frequency selection choice is provided in this app.

        ''')

