from flask import Flask, request, jsonify
import joblib
import numpy as np
import threading
import time
import requests

app= Flask(__name__)
model=joblib.load('iris_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features=np.array(data['features']).reshape(1,-1)
    prediction=model.predict(features)[0]
    return jsonify({'prediction': int(prediction)})

def run_flask():
    app.run(debug=False)

flask_thread=threading.Thread(target=run_flask)
flask_thread.daemon=True
flask_thread.start()

time.sleep(2)

# MAking a Test Request
url="http://127.0.0.1:5000/predict"
data={'features':[5.1,3.4,1.4,0.2]}
response=requests.post(url, json=data)
print("Response from API:",response.json())
