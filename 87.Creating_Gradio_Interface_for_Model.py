import gradio as gr
import joblib 
import numpy as np

model = joblib.load('iris_model.pkl')

def predict_iris(sepal_l, sepal_w, petal_l, petal_w):
    features= np.array([sepal_l, sepal_w, petal_l, petal_w]).reshape(1,-1)
    pred =model.predict(features)[0]
    return "Setosa" if pred == 1 else "Not Setosa"

gr.Interface(
    fn=predict_iris,
    inputs=["number", "number", "number", "number"],
    outputs="text",
    title="Iris Prediction"
).launch()
