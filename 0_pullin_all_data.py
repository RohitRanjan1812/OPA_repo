#%%
import pandas as pd
import os
import yfinance as yf

directory = r'C:\Users\49176\Desktop\DSA\OPA_repo\Data\csv'
sp_data_ticker = list()
for f in os.listdir(directory):
    f_path = os.path.join(directory, f)
    if os.path.isfile(f_path):
        sp_data_ticker.append(f[:-4])
        #print(f_path)
        #print(sp_data_ticker)
yf_ticker_df = pd.read_excel(r'C:\Users\49176\Desktop\DSA\OPA_repo\Data\tickers_usa.ods', engine='odf')
yf_ticker = list(yf_ticker_df.Ticker)

print('check if all data tickers are in yfinace ticker list:', set(sp_data_ticker) - set(yf_ticker) == set())

#%%
df_allInfo = pd.DataFrame([])
for t in sp_data_ticker:
    temp_df = pd.DataFrame.from_dict([yf.Ticker(t).info]) 
    df_allInfo = pd.concat([df_allInfo, temp_df])

df_allInfo.to_pickle(r'C:\Users\49176\Desktop\DSA\OPA_repo\df_allInfo_124.pkl')

#%%
#for handling incompatible inputs
issue_tickers = []
for t in list(set(sp_data_ticker) - set(df_allInfo.symbol)):
    try:
        temp_df = pd.DataFrame.from_dict([yf.Ticker(t).info]) 
        df_allInfo = pd.concat([df_allInfo, temp_df])
    except:
        issue_tickers.append(t)
        continue

#['REGN', 'TJX', 'TRAUF'] are the issue tickers -> we will manually append them
df_allInfo = pd.concat([df_allInfo, pd.DataFrame.from_dict([yf.Ticker('REGN').info])])
df_allInfo = pd.concat([df_allInfo, pd.DataFrame.from_dict([yf.Ticker('TJX').info])])
df_allInfo = pd.concat([df_allInfo, pd.DataFrame.from_dict([yf.Ticker('TRAUF').info])])
#the manual append works fine, api call times could have been an issue why it threw an exception

#%% finally we can save our complete extract as pkl
df_allInfo.to_pickle(r'C:\Users\49176\Desktop\DSA\OPA_repo\df_allInfo.pkl')

#%%download all 503 stock ticker data via api call
df = pd.read_pickle(r'C:\Users\49176\Desktop\DSA\OPA_repo\df_allInfo_clean.pkl')
ticker_str = " ".join(list(df.index))
data = yf.download(ticker_str, start="1900-01-01",  #max 50 yrs data is only available so 1962-01-02
                    end="2022-12-10", group_by='tickers')
data.to_pickle('50yr_timeSeries_data.pkl')

