# train_model.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pickle
import os

# Load the dataset (replace 'your_dataset.csv' with your actual dataset file)
data = pd.read_csv('your_dataset.csv')

# Preprocess the data
X_data = data[['humidity', 'wind_speed', 'temperature']].values  # Features as numpy array
y_data = data['AQI'].values  # Target as numpy array

# Scale features and target
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

X_scaled = scaler_X.fit_transform(X_data)
y_scaled = scaler_y.fit_transform(y_data.reshape(-1, 1))

# Create sequences for LSTM
n_steps = 60  # Number of time steps to consider for each input
X, y = [], []

for i in range(n_steps, len(X_scaled)):
    X.append(X_scaled[i - n_steps:i])  # Previous 60 steps
    y.append(y_scaled[i])  # Corresponding target value

X, y = np.array(X), np.array(y)

# Define LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)  # Output layer for regression
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2)

# Save the LSTM model
if not os.path.exists('models'):
    os.makedirs('models')

model.save('models/lstm_aqi_model.h5')

# Save the scalers for later use
with open('models/scaler_X.pkl', 'wb') as f:
    pickle.dump(scaler_X, f)

with open('models/scaler_y.pkl', 'wb') as f:
    pickle.dump(scaler_y, f)

print("LSTM model training complete and saved to 'models/lstm_aqi_model.h5'")
print("Scalers saved to 'models/scaler_X.pkl' and 'models/scaler_y.pkl'")
