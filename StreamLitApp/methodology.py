import streamlit as st
from PIL import Image


def _load_img():
    coll = dict()
    coll['hcp_dend'] = Image.open(r'D:\Py_Prjs\OPA_repo\data\images\hca_dendogram.png')
    coll['knn_opt'] = Image.open(r'D:\Py_Prjs\OPA_repo\data\images\knn_optimalCluster.png')
    coll['knn_bestfeatures'] = Image.open(r'D:\Py_Prjs\OPA_repo\data\images\knn_onSelectedFeatures.png')
    return coll
coll = _load_img()
@st.cache_data
def docu():
    st.write('This tab consists of results from the clustering analysis and then advices the user to select one out of the 14 alternative portfolios')
    st.write('''
            We first look at the Hierarchical  Clustering Analysis
            ''')

    st.image(coll['hcp_dend'])
    st.image(coll['knn_opt'])
    st.image(coll['knn_bestfeatures'])