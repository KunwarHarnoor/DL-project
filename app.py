# app.py
from flask import Flask, request, jsonify
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model, with error handling if it doesn’t exist or is corrupted
try:
    model_path = 'models/trained_model.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully.")
    else:
        raise FileNotFoundError("Model file not found. Please ensure 'trained_model.pkl' exists in the 'models' directory.")
except Exception as e:
    raise Exception(f"Failed to load the model. Error: {str(e)}")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Validate data
    required_keys = {'humidity', 'wind_speed', 'temperature'}
    if not all(key in data for key in required_keys):
        return jsonify({'error': 'Missing required data'}), 400

    # Convert the data to a DataFrame
    new_data = pd.DataFrame([data])

    # Make prediction
    prediction = model.predict(new_data)
    
    # Return the result
    return jsonify({'predicted_AQI': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
    
