import pandas as pd
from flask import Flask, request
import pickle
import json

app = Flask(__name__)
model = None

@app.route('/')
def health_check():
    return "Service is running"

@app.route('/predict', methods=['GET', 'POST'])
def predict():

    model = pickle.load(open('artifacts/model.pkl', 'rb'))

    if request.method == 'POST':
        df = pd.read_json(request.data, orient='split')
        return {"predictions":model.predict(df).tolist()}
    else:
        return "Please POST data to this endpoint to return predictions"
