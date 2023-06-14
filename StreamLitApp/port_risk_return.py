import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

alt_port_path = r'D:\Py_Prjs\OPA_repo\data\alternate_port.pkl'
wt_port_path = r'D:\Py_Prjs\OPA_repo\data\dict_alt_port_mlp_weight.pkl'
base_path = r'D:\Py_Prjs\OPA_repo\data\\'

df_alt_port = pd.read_pickle(alt_port_path)
df_wt_port = pd.read_pickle(wt_port_path)
def plot_all_risk_return(df, current_port, mod, resamp):

    # Create a color map
    cmap = plt.get_cmap('jet') # type: ignore

    # Create a normalize object to map index values to the range [0, 1]
    norm = plt.Normalize(vmin=0, vmax=len(df['index'].unique()) - 1) # type: ignore
    fig, ax = plt.subplots(figsize=(12, 6))
    # Scatter plot with beta on x-axis and return on y-axis
    for i, category in enumerate(df['index'].unique()):
        df_category = df[df['index'] == category]
        ax.scatter(df_category['beta'], df_category[f'return_{mod}_{resamp}'], color=cmap(norm(i)), label=category)
    
    scale = 1
    if mod == 'pt':
        scale = 2.3*scale

    # Label the points by 'index' and make a circler around the current one
    for i in range(len(df)):
        if i + 1 == current_port:
            circle = Ellipse((df['beta'][i], df[f'return_{mod}_{resamp}'][i]), 0.002, 0.33*scale, facecolor='none',
                    edgecolor=(1, 0, 0), linewidth=1.5, alpha=1)
            ax.add_patch(circle)
        plt.text(df['beta'][i] + 0.001, df[f'return_{mod}_{resamp}'][i] + 0.01, df['index'][i].split(' ')[-1])

    ax.set_xlabel('Beta')
    ax.set_ylabel('Return')
    # Move the legend to an empty part of the plot
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    fig.savefig(base_path + f'risk_return_alt_port_{current_port}_{mod}_{resamp}.jpg', dpi=100, bbox_inches='tight')
    plt.close()

portfolio_info = dict()
for i in range(14):
    temp = df_wt_port[f'alt_port_{i}'].join(df_alt_port[f'cluster_{i}'][['beta']])
    temp['prod'] = temp['weight']*temp['beta']
    portfolio_info[f'Alternative portfolio {i+1}'] = {'beta': temp['prod'].sum()}

    for mod in ['ts', 'pt']:
        for resamp in ['1W', '1M']:
            if mod == 'ts':
                    if resamp == '1M':
                        file_name = f'ts_model_alt_port_{i}_1M.jpg'
                        model = pickle.load(open(base_path + file_name[:-4] + '.sav', 'rb'))
                        portfolio_info[f'Alternative portfolio {i+1}'][f'return_ub_{mod}_{resamp}'] = 100*(np.exp(model.get_forecast(steps = 12).summary_frame()).iloc[-1, 3] -1000) /1000
                        portfolio_info[f'Alternative portfolio {i+1}'][f'return_{mod}_{resamp}'] = 100*(np.exp(model.get_forecast(steps = 12).summary_frame()).iloc[-1, 0] -1000) /1000
                        portfolio_info[f'Alternative portfolio {i+1}'][f'return_lb_{mod}_{resamp}'] = 100*(np.exp(model.get_forecast(steps = 12).summary_frame()).iloc[-1, 2] -1000) /1000
                    else:
                        file_name = f'ts_model_alt_port_{i}_1W.jpg'
                        model = pickle.load(open(base_path + file_name[:-4] + '.sav', 'rb'))
                        portfolio_info[f'Alternative portfolio {i+1}'][f'return_ub_{mod}_{resamp}'] = 100*(np.exp(model.get_forecast(steps = 52).summary_frame()).iloc[-1, 3] -1000) /1000
                        portfolio_info[f'Alternative portfolio {i+1}'][f'return_{mod}_{resamp}'] = 100*(np.exp(model.get_forecast(steps = 52).summary_frame()).iloc[-1, 0] -1000) /1000
                        portfolio_info[f'Alternative portfolio {i+1}'][f'return_lb_{mod}_{resamp}'] = 100*(np.exp(model.get_forecast(steps = 52).summary_frame()).iloc[-1, 2] -1000) /1000
            else:
                if resamp == '1M':
                    file_name = f'pt_model_alt_port_{i}_1M.jpg'
                    model = pickle.load(open(base_path + file_name[:-4] + '.sav', 'rb'))
                    # prediction DF
                    future_dates = model.make_future_dataframe(periods = 12, freq = 'MS')

                    # prediction
                    forecast = model.predict(future_dates)
                    portfolio_info[f'Alternative portfolio {i+1}'][f'return_ub_{mod}_{resamp}'] = 100*(forecast.iloc[-1, 3] -1000)/1000
                    portfolio_info[f'Alternative portfolio {i+1}'][f'return_{mod}_{resamp}'] = 100*(forecast.iloc[-1, -1] -1000)/1000
                    portfolio_info[f'Alternative portfolio {i+1}'][f'return_lb_{mod}_{resamp}'] = 100*(forecast.iloc[-1, 2] -1000)/1000

                else:
                    file_name = f'pt_model_alt_port_{i}_1W.jpg'
                    model = pickle.load(open(base_path + file_name[:-4] + '.sav', 'rb'))  
                    # prediction DF
                    future_dates = model.make_future_dataframe(periods = 52, freq = 'W') 

                    # prediction
                    forecast = model.predict(future_dates)
                    portfolio_info[f'Alternative portfolio {i+1}'][f'return_ub_{mod}_{resamp}'] = 100*(forecast.iloc[-1, 3] -1000)/1000
                    portfolio_info[f'Alternative portfolio {i+1}'][f'return_{mod}_{resamp}'] = 100*(forecast.iloc[-1, -1] -1000)/1000
                    portfolio_info[f'Alternative portfolio {i+1}'][f'return_lb_{mod}_{resamp}'] = 100*(forecast.iloc[-1, 2] -1000)/1000

port_info_df = pd.DataFrame(portfolio_info).T.reset_index()

with open(base_path + r'\dict_alt_port_risk_return_info.pkl' ,'wb') as handle:
    pickle.dump(port_info_df ,handle ,protocol=pickle.HIGHEST_PROTOCOL)

for i in range(1, 15):
    for mod in ['ts', 'pt']:
        for resamp in ['1W', '1M']:
            df = port_info_df[['index', 'beta', f'return_{mod}_{resamp}']]
            plot_all_risk_return(df, i, mod, resamp)