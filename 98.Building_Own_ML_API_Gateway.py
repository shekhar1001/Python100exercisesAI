from flask import Flask, request, jsonify
import requests

app=Flask(__name__)

@app.route('/gateway',methods=['POST'])
def gateway():
    payload=request.get_json()
    response=requests.post('http://model-service:5000/predict', json=payload)
    return jsonify(response.json())

if __name__=='__main__':
    app.run(debug=True)

    