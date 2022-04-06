import streamlit as st
from utils import list_models, get_all_preds_and_urls, get_least_conf_inputs, pred_and_create_dfs
from PIL import Image
import pandas as pd
import requests
from io import BytesIO
import grpc

# create main portion of app
def main():
    st.set_page_config(layout="wide")
########## Step 1: connecting to Clarifai ############

    st.markdown("""
# Classifier Uncertainty Explorer
This is a Streamlit app that surfaces the difficult examples that your visual classifier
 is least certain about.

---
### Step 1: Enter API key for app
    """)

    api_key = st.text_input("Enter API Key, then press enter:", value = 'cf375e582d214f51a72d88385a6a7205')

    if api_key == '' and  'api_key' not in st.session_state:
        st.warning("App Key is not entered")
        st.stop()
    else:
        st.session_state['api_key'] = api_key
        st.write("App Key: {}".format(api_key))

########## Step 2: listing models ############
    st.markdown("""
### Step 2: Select a model
    """)

    model_list = [''] + list_models(api_key)

    model_id = st.selectbox('Select a model', model_list)

    if model_id == '' and 'model_id' not in st.session_state:
        st.warning("Select a model")
        st.stop()
    else:
        st.session_state['model_id'] = model_id
        st.write("Model ID: {}".format(model_id))

########## Step 3: retrieving predictions ############
    st.markdown("""
### Step 3: Retrieve predictions...
    """)

    preds_df = None
    urls = None
    least_conf_input_ids = None
    with st.spinner('Retrieving predictions on all inputs...'):
        preds_df, urls, least_conf_input_ids = pred_and_create_dfs(api_key, model_id)

    st.text('Done!')

    
########## Step 4: displaying predictions ############
    st.markdown("""
### Step 4: Display predictions
    """)

    start_ind = st.number_input('Start index', value=0)
    end_ind = st.number_input('End index', value=10)

    for i in range(start_ind, end_ind):
        with st.container():
            input_id = least_conf_input_ids[i]

            input_preds = preds_df[preds_df['input_id'] == input_id]
            ground_truth = input_preds.iloc[0]['ground_truth']

            preds = input_preds[['pred', 'confidence']]

            col1, col2, col3 = st.columns(3)
            col1.subheader("Image")
            col1.image(urls[input_id])
            
            col2.subheader("Ground Truth")
            col2.text(ground_truth)

            col3.subheader("Predictions")
            col3.dataframe(preds)

if __name__ == '__main__':
    main()
