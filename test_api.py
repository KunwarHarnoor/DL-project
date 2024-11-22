# test_api.py
import requests

# Define the API endpoint URL
url = "http://127.0.0.1:5000/predict"

# Example data (replace these values with actual inputs)
data = {
    'humidity': 85,
    'wind_speed': 6,
    'temperature':16 
}

# Send the POST request
response = requests.post(url, json=data)

# Print the response
print(response.json())
