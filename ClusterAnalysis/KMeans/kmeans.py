import pandas as pd
df = pd.read_pickle(r'D:\Py_Prjs\OPA_repo\ClusterAnalysis\KMeans\Data\df_allInfo_clean.pkl')
df_numeric = df.select_dtypes(include=['float64'])