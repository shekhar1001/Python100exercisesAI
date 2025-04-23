from flask import Flask, request,jsonify
import joblib
import numpy as np

app=Flask(__name__)
model=joblib.load('iris_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data=request.get_json()
    features=np.array(data['features']).reshape(1,-1)
    prediction=model.predict(features)[0]
    return jsonify({'prediction':omt(prediction)})

if __name__=='__main__':
    app.run(debug=True)


