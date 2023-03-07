import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from pandas.io.clipboards import read_clipboard
st.write("""
# Sales Prediction App

This app predicts the **Sales** !
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    TV = st.sidebar.slider('TV', 0.7, 296.4, 100)
    Radio = st.sidebar.slider('Radio', 0.0, 49.6, 30)
    Newspaper = st.sidebar.slider('Newspaper', 0.3, 114, 30.5)    
    data = {'TV': TV,
            'Radio': Radio,
            'Newspaper': Newspaper}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

data = pd.read_csv('Advertising.csv')

X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

model = LinearRegression
model.fit(X, y)


prediction = model.predict(df)

st.subheader('Prediction')
st.write(prediction)
