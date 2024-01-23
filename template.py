# Streamlit & Data Handling
import streamlit as st
import pandas as pd
import altair as alt
from io import BytesIO
import datetime

# Machine Learning & Evaluation
from sklearn.base import BaseEstimator
import pickle
from sklearn.metrics import mean_absolute_error

# Communcation & Visualization
import requests
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
from PIL import Image

im = Image.open('favicon-32x32.png')
st.set_page_config(
    layout="wide",
    page_title='ALVA x Purwadhika Workshop')


if __name__ == '__main__':
    with st.sidebar.container():
        selected_page = option_menu(
            menu_title='ALVA X Purwadhika Workshop',
            options=['Case 1','Case 2'],
            styles={
                "container": {
                        "max-width": "700px"
                    },
                "nav-link-selected": {"background-color": "#32b280"}, ##7605c1
            }
        )

    if selected_page == 'Case 1':
        st.title('Start your page here')
    else:
        st.title('Start your page here')