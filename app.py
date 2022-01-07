# pip install streamlit
import streamlit as st
import numpy as np
from model import post_new
st.set_page_config(page_title="Reddit Up votes  Prediction App",
                   layout="wide")


with st.form("prediction_form"):

    st.header("Enter the URL of Reddit post:")

    col1,col2 = st.beta_columns(2)
    with col2:
        text1= st.text_input('URL link to scrape')

        st.write('the link:')
        st.write(text1)
    
    submit_val = st.form_submit_button("Predict Upvotes")

if submit_val:
    # If submit is pressed == True
   


    #if attribute.shape == (1,20):
        #print("attributes valid")
        

    value = post_new(text1)


    st.header("Here are the results:")
    st.success(f"The upvotes predicted is {value} ")