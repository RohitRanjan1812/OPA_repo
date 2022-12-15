#%%
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import yahoo_fin as yfin
import yfinance as yf

#%%
df = pd.read_pickle(r'C:\Users\49176\Desktop\DSA\OPA_repo\df_allInfo.pkl')

#%% clean null values -- some symbols dropped --> {'CTXS', 'KACPF', 'NLSN'}
df = df.dropna(subset='symbol', axis=0)

#%% drop cols which are empty
cols_by_null_values = (df.isna().sum()*100/len(df)).sort_values(ascending=False)
df = df.drop(list(cols_by_null_values[cols_by_null_values.values == 100].index), axis=1)

#%%remaining cols over 99% empty
rem_cols_by_null_value = (df.isna().sum()*100/len(df)).sort_values(ascending=False)
rem_cols_gt_99 = rem_cols_by_null_value[rem_cols_by_null_value.values > 99].index
for col in list(rem_cols_gt_99):
    print(df[col].value_counts())

#%% doesn seem much useful so we can drop these cols as well
df = df.drop(list(rem_cols_gt_99), axis=1)
rem_cols_by_null_value = (df.isna().sum()*100/len(df)).sort_values(ascending=False)

#%%insuring we have the most up to date S&P tickers using yahoo_fin lib
from yahoo_fin import stock_info as si
sp500_tickers = si.tickers_sp500(True)
#%%
sp_data_ticker = list(sp500_tickers.Symbol.values)
df_allInfo = pd.DataFrame([])
issue_tickers = []
for t in list(set(sp_data_ticker) - set(df.symbol)):
    try:
        temp_df = pd.DataFrame.from_dict([yf.Ticker(t).info]) 
        df_allInfo = pd.concat([df_allInfo, temp_df])
    except:
        issue_tickers.append(t)
        continue
#%%filter out out dated/removed tickers
df = df.merge(sp500_tickers.Symbol, left_on='symbol', right_on='Symbol', how='inner')\
       .drop('Symbol', axis=1)
#add new tickers currently in S&P500 via yahoo_fin lib
df = pd.concat([df, df_allInfo]) 

#%%clean null columns again
cols_by_null_values = (df.isna().sum()*100/len(df)).sort_values(ascending=False)
df = df.drop(list(cols_by_null_values[cols_by_null_values.values == 100].index), axis=1)

#%%use the describe and value_count methods on all the remaining 127 cols to assess if we need to keep it
col = 'heldPercentInstitutions'
print(rem_cols_by_null_value[col])
print(df[col].describe())
print(df[col].value_counts())

#%%cols to keep following assessment
cols_keep = [#'fax',
             #'address2',
             #'uuid',
             'trailingPegRatio',
             #'lastDividendDate',
             'lastDividendValue',
             #'preMarketPrice',
             #'impliedSharesOutstanding',
             '52WeekChange',
             'SandP52WeekChange',
             'fiveYearAvgDividendYield',
             #'lastSplitDate',
             'lastSplitFactor',
             'dividendRate',
             'dividendYield',
             #'exDividendDate',
             'earningsGrowth',
             'earningsQuarterlyGrowth',
             'debtToEquity',
             'freeCashflow',
             'trailingPE',
             'trailingAnnualDividendYield',
             'trailingAnnualDividendRate',
             'ebitda',
             'enterpriseToEbitda',
             'priceToBook',
             'returnOnEquity',
             'currentRatio',
             'quickRatio',
             'operatingCashflow',
             'state',
             'beta',
             'targetMeanPrice',
             'targetMedianPrice',
             'targetHighPrice',
             'numberOfAnalystOpinions',
             'targetLowPrice',
             'recommendationMean',
             'payoutRatio',
             'priceToSalesTrailing12Months',
             'returnOnAssets',
             'pegRatio',
             'totalCashPerShare',
             'totalRevenue',
             'revenueGrowth',
             'enterpriseToRevenue',
             'totalDebt',
             'floatShares',
             'totalCash',
             'netIncomeToCommon',
             'marketCap',
             'fullTimeEmployees',
             'forwardPE',
             'trailingEps',
             'forwardEps',
             'sharesOutstanding',
             'bookValue',
             'shortPercentOfFloat',
             #'phone',
             'twoHundredDayAverage',
             #'previousClose',
             #'regularMarketOpen',
             #'tradeable',
             #'dayHigh',
             #'regularMarketPrice',
             #'ask',
             #'bidSize',
             #'averageVolume',
             #'bid',
             'fiftyTwoWeekLow',
             #'regularMarketDayHigh',
             #'regularMarketPreviousClose',
             'fiftyDayAverage',
             'fiftyTwoWeekHigh',
             #'open',
             #'averageVolume10days',
             #'volume',
             #'regularMarketDayLow',
             'currency',
             #'askSize',
             #'regularMarketVolume',
             #'dayLow',
             #'averageDailyVolume10Day',
             #'zip',
             #'dateShortInterest',
             'ebitdaMargins',
             'revenuePerShare',
             #'financialCurrency',
             #'currentPrice',
             'grossProfits',
             'recommendationKey',
             'operatingMargins',
             'grossMargins',
             'profitMargins',
             'industry',
             'shortName',
             #'address1',
             #'maxAge',
             #'website',
             #'companyOfficers',
             'country',
             #'logo_url',
             'city',
             #'longBusinessSummary',
             'exchange',
             #'longName',
             #'priceHint',
             #'lastFiscalYearEnd',
             'enterpriseValue',
             'sharesShortPriorMonth',
             #'sharesShortPreviousMonthDate',
             'shortRatio',
             #'mostRecentQuarter',
             #'nextFiscalYearEnd',
             'heldPercentInsiders',
             'sector',
             'sharesPercentSharesOut',
             #'exchangeTimezoneName',
             'sharesShort',
             #'market',
             #'messageBoardId',
             'symbol',
             #'quoteType',
             #'gmtOffSetMilliseconds',
             #'isEsgPopulated',
             #'exchangeTimezoneShortName',
             'heldPercentInstitutions']
#%%filter our df for the interesting columns and sort them from left to right by % availability 
df = df[rem_cols_by_null_value[cols_keep].sort_values(ascending=True).index].set_index('symbol')

#%%save it as pickle
df.to_pickle('df_allInfo_clean.pkl')