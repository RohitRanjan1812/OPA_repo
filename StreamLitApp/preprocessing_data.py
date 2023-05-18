import pandas as pd

def fix_tickerInfo_dtypes(df:pd.DataFrame):
    df_obj = df.reset_index().select_dtypes(include='object')
    real_obj_cols = ['symbol', 'recommendationKey', 'industry', 'shortName', 'country',
        'exchange', 'city', 'sector', 'currency', 
        #'sharesOutstanding',
        #'totalDebt', 'marketCap', 'netIncomeToCommon', 'totalCash',
        #'floatShares', 'totalRevenue', 
        'numberOfAnalystOpinions', 'state',
        #,
        #'operatingCashflow', 'priceToBook', 'ebitda', 'trailingPE',
        #'freeCashflow', 'dividendYield', 'dividendRate', 
        'lastSplitFactor'#,
        #'fiveYearAvgDividendYield', 'lastDividendValue', 'trailingPegRatio'
        ]
    cols_to_convert = list(set(df_obj.columns) - set(real_obj_cols))

    df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors='coerce')
    return df

def get_tickers_metrics_ts(ts_df):
    l1, l2 = zip(*ts_df.columns)
    tickers = list(set(l1))
    metrics = list(set(l2))
    return tickers, metrics

def get_timeSeries(ts_df, ticker, metric):
    return pd.DataFrame(ts_df[:][ticker][metric]).reset_index().rename(columns={'Date': 'date', metric: 'value'})