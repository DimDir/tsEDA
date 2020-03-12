import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import datetime

st.title("Time Series EDA")

# data_URL = "D:\Data\Metro_Interstate_Traffic_Volume.csv"
# @st.cache
# def load_data(nrows):
#     data = pd.read_csv(data_URL, nrows=nrows)
#     return data

path = st.sidebar.file_uploader('DOWNLOAD FILE', type='csv')

eda_steps = st.sidebar.radio('Choose the options:', ('Preprocessing', 'Models'))

# data_load_state = st.text('Loading data...')
# data = load_data(2000)
# data_load_state = st.text('Loading data has done!')

if path is not None:
    data = st.cache(pd.read_csv)(path)
    if eda_steps == 'Preprocessing':

        ## Show data
        st.table(data.head())
        st.write(data.shape)


        column = st.radio("Choose the Time series", (data.columns))
        data[column] = pd.to_datetime(data[column])
        data2 = data.set_index(column)
        st.write(data2.head(12))

    if st.button('Create plot'):
        target = 'traffic_volume'
        #st.line_chart(data2['traffic_volume'])
        # plt.plot(data2['traffic_volume'][:1000])
        # st.pyplot()
        result = sm.tsa.seasonal_decompose(data2[target], model='addective', freq=24*30*12)
        result.plot()
        plt.tick_params(labelrotation=45)
        st.pyplot()

