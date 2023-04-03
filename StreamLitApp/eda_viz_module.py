import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import streamlit as st 

df = pd.read_pickle(r'D:\Py_Prjs\OPA_repo\data\df_allInfo_clean.pkl')
def test_graph(df):
    width = st.sidebar.slider("plot width", 1, 25, 3)
    height = st.sidebar.slider("plot height", 1, 25, 1)
    sns.set()
    fig, (ax_1, ax_2) = plt.subplots(2, 1, figsize=(width, height), facecolor='lightskyblue',
                        layout='constrained')
    # fig.suptitle('Figure', fontsize=10)
    # ax_1.set_title('Axes', loc='left', fontstyle='oblique', fontsize=7)
    # ax_1.set_ylabel('Count', fontsize=7)
    # ax_1.set_yticklabels([i for i in range(0, 100, 20)], fontsize=7)
    # ax_1.set_xlabel('held % Institutions')
    # # s = [label.get_text() for label in ax_1.get_xticklabels()]
    # # ax_1.set_xticklabels(s, fontsize=7)
    sns.histplot(data=df, x='heldPercentInstitutions', ax=ax_1)
    sns.histplot(data=df, x='sector', ax=ax_2)
    plt.xticks(rotation=90)
    st.pyplot(fig)
    # ax_2 = fig.add_subplot(1, 2, 2)
    # st.pyplot()