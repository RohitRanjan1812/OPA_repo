import pandas as pd
import yfinance as yf
df = pd.read_excel("tickers.ods", engine="odf")
# print(df.head())
tickers = list(df.Tickers)
df_allInfo = pd.DataFrame([])
count = 0
for t in tickers:
    if count == 100:
        break
    temp_df = pd.DataFrame.from_dict([yf.Ticker(t).info]) 
    df_allInfo = pd.concat([df_allInfo, temp_df])
    count = count + 1

print(df_allInfo.head())