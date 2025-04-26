from fastapi import FastAPI
from pydantic import BaseModel
import joblib 
import numpy as np

class IrisInput(BaseModel):
    features: list

app= FastAPI()
model=joblib.load('iris_model.pkl')

@app.post("/predict")
def predict(data: IrisInput):
    features = np.array(data.features).reshape(1,-1)
    prediction=model.predict(features)[0]
    return {"prediction":int(prediction)}