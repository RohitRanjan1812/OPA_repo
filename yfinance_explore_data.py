import pandas as pd
import yfinance as yf
df = pd.read_excel("tickers.ods", engine="odf")
# print(df.head())
tickers = list(df.Tickers)

# too big df due to changing columns
# df_allInfo = pd.DataFrame([])
# for t in tickers:
#     temp_df = pd.DataFrame.from_dict([yf.Ticker(t).info]) 
#     df_allInfo = pd.concat([df_allInfo, temp_df])

# print(df_allInfo.head())

#find all distinct columns
cols = set()
for t in tickers[:100]:
    cols = cols.union(set(yf.Ticker(t).info.keys()))

print('xyz')
print('hello waldo')