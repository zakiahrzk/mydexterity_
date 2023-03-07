import streamlit as st
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

st.write("""
# Sales Prediction App

This app predicts the **Sales** using Linear Regression!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    TV = st.sidebar.slider('TV', 10, 300, 150)
    Radio = st.sidebar.slider('Radio', 0, 50, 25)
    Newspaper = st.sidebar.slider('Newspaper', 0, 120, 50)    
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

model = LinearRegression()
model.fit(X, y)

prediction = model.predict(df)

st.subheader('The sales prediction is')
st.write(prediction)
