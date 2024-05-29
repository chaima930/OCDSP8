
from flask import Flask, request, jsonify
import numpy as np
import pickle
from sklearn.preprocessing import  MinMaxScaler
from lightgbm import LGBMClassifier

app = Flask(__name__)

#load models
with open('data/model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('data/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

@app.route('/')
def home_page():
    return 'Welcome to the credit scoring api'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from POSTrequest
        data = request.get_json(force=True)
        df_test = np.array(data['df_test'])

        # Scaling
        scaled_data = scaler.transform(df_test)

        # prediction
        prediction = model.predict_proba(scaled_data)[:, 1]  # Assuming a binary classification task

        # Send predictions in JSON format
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        # handle potential issues
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # run app in debug
    app.run(port='5003')
