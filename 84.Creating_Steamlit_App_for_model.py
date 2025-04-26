import streamlit as st
import joblib 
import numpy as np

model=joblib.load('iris_model.pkl')

st.title("Iris Classifier")
sepal_length=st.slider('Sepal Length', 4.0, 8.0)
sepal_width = st.slider('Sepal Width', 2.0, 4.5)
petal_length = st.slider('Petal Length', 1.0, 7.0)
petal_width = st.slider('Petal Width', 0.1, 2.5)

features=np.array([sepal_length, sepal_width, petal_length,petal_width])
prediction= model.predict(features)[0]

st.write(f"Prediction:{'Setosa' if prediction ==1 else "Not Setosa"}")