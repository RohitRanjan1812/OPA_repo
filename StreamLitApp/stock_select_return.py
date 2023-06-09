import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import numpy as np

path = r'D:\Py_Prjs\OPA_repo\data\dict_alt_port_mlp_weight.pkl'
base_path = r'D:\Py_Prjs\OPA_repo\data\\'
df_wt = pd.read_pickle(path)
for k in df_wt.keys():
    x = df_wt[k].reset_index()
    x = x.rename(columns={'index': 'Stock'})
    df_wt[k] = x
pd.set_option('display.max_colwidth', 1000)

def disp_img(file_name):
    st.image(Image.open(base_path + file_name))

def main():
    state = st.session_state
    options = [f'Alternative Portfolio {i}' for i in range(1, 15)]
    default_port = options[0]
    model_options = ['SARIMA', 'PROPHET']
    default_mod_opt = model_options[0]
    resampling_options = ['1 Month', '1 Week']
    default_resamp_opt = resampling_options[0]
    
    # Define variables to store dropdown values
    port_val = state.port_val if "port_val" in state else default_port
    mod_val = state.mod_val if "mod_val" in state else default_mod_opt
    resamp_val = state.resamp_val if "resamp_val" in state else default_resamp_opt
    
    col1, col2 = st.columns([1.2, 5])
    with col1:
        port_val = st.selectbox('Select an alternative portfolio option', options, index=options.index(port_val))
        state.port_val = port_val
        alt_num = int(port_val.split(' ')[-1]) - 1
        st.write('Here are the weight distribution for:', port_val)
        st.table(df_wt[f"alt_port_{alt_num}"].to_dict())
    with col2:
        col21, col22 = st.columns(2)
        with col21:
            mod_val = st.selectbox('Select the prediction method:', model_options, index=model_options.index(mod_val))
            state.mod_val = mod_val
            resamp_val = st.selectbox('Select the sampling freq:', resampling_options, index=resampling_options.index(resamp_val))
            state.resamp_val = resamp_val
            st.write('selected model option', mod_val)
            st.write('selected resampling option', resamp_val)
            if mod_val == 'SARIMA':
                if resamp_val == '1 Month':
                    file_name = f'ts_model_alt_port_{alt_num}_1M.jpg'
                    model = pickle.load(open(base_path + file_name[:-4] + '.sav', 'rb'))
                    return_pct_ub = 100*(np.exp(model.get_forecast(steps = 12).summary_frame()).iloc[-1, 3] -1000) /1000
                    return_pct = 100*(np.exp(model.get_forecast(steps = 12).summary_frame()).iloc[-1, 0] -1000) /1000
                    return_pct_lb = 100*(np.exp(model.get_forecast(steps = 12).summary_frame()).iloc[-1, 2] -1000) /1000
                else:
                    file_name = f'ts_model_alt_port_{alt_num}_1W.jpg'
                    model = pickle.load(open(base_path + file_name[:-4] + '.sav', 'rb'))
                    return_pct_ub = 100*(np.exp(model.get_forecast(steps = 52).summary_frame()).iloc[-1, 3] -1000) /1000
                    return_pct = 100*(np.exp(model.get_forecast(steps = 52).summary_frame()).iloc[-1, 0] -1000) /1000
                    return_pct_lb = 100*(np.exp(model.get_forecast(steps = 52).summary_frame()).iloc[-1, 2] -1000) /1000
            else:
                if resamp_val == '1 Month':
                    file_name = f'pt_model_alt_port_{alt_num}_1M.jpg'
                    model = pickle.load(open(base_path + file_name[:-4] + '.sav', 'rb'))
                    # prediction DF
                    future_dates = model.make_future_dataframe(periods = 12, freq = 'MS')

                    # prediction
                    forecast = model.predict(future_dates)
                    return_pct_ub = 100*(forecast.iloc[-1, 3] -1000)/1000
                    return_pct = 100*(forecast.iloc[-1, -1] -1000)/1000
                    return_pct_lb = 100*(forecast.iloc[-1, 2] -1000)/1000

                else:
                    file_name = f'pt_model_alt_port_{alt_num}_1W.jpg'
                    model = pickle.load(open(base_path + file_name[:-4] + '.sav', 'rb'))  
                    # prediction DF
                    future_dates = model.make_future_dataframe(periods = 52, freq = 'W') 

                    # prediction
                    forecast = model.predict(future_dates)
                    return_pct_ub = 100*(forecast.iloc[-1, 3] -1000)/1000
                    return_pct = 100*(forecast.iloc[-1, -1] -1000)/1000
                    return_pct_lb = 100*(forecast.iloc[-1, 2] -1000)/1000
            disp_img(file_name)

        with col22:
            if mod_val == 'SARIMA':
                st.write('Predicted 1 year highest expected % return :')
                st.write(return_pct_ub)
                st.write('Predicted 1 year expected % return :')
                st.write(return_pct)
                st.write('Predicted 1 year lowest expected % return :')
                st.write(return_pct_lb)
                st.write(model.summary())
            else:
                st.write('Predicted 1 year highest expected % return :')
                st.write(return_pct_ub)
                st.write('Predicted 1 year expected % return :')
                st.write(return_pct)
                st.write('Predicted 1 year lowest expected % return :')
                st.write(return_pct_lb)
                st.write(model.plot_components(forecast))             



    # # Sum all values for calculating percentage
    # df = df_0['alt_port_0'][-1:]


    # # Sum all values for calculating percentage
    # total = df.sum(axis=1).values[0]

    # # Prepare data for the pie chart
    # labels = df.columns.tolist()
    # values = df.values[0].tolist()

    # # Create a dictionary from labels and values
    # data_dict = dict(zip(labels, values))

    # # Sort the dictionary by value in descending order and create separate lists for top 26 and the rest
    # sorted_dict = dict(sorted(data_dict.items(), key=lambda item: item[1], reverse=True))
    # top_26_labels = list(sorted_dict.keys())[:26]
    # top_26_values = list(sorted_dict.values())[:26]
    # other_values = list(sorted_dict.values())[26:]

    # # Add 'Other' category
    # top_26_labels.append('Other')
    # top_26_values.append(sum(other_values))

    # # Calculate percentage
    # sizes = [(value / total) * 100 for value in top_26_values]

    # # fig, ax = plt.subplots(figsize=(2,2))  # Specify the figure size here
    # plt.pie(sizes, labels=top_26_labels, autopct='%1.1f%%', startangle=90, pctdistance=0.85, wedgeprops=dict(width=0.5))
    # # ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # st.pyplot(fig)
