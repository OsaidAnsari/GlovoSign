import requests
import joblib
import numpy as np

model_filename = 'decision_tree_model.joblib'
clf = joblib.load(model_filename)

def get_sensor_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

nodemcu_url = "http://192.168.100.21:100/"  # Replace with the actual URL
sensor_data = get_sensor_data(nodemcu_url)

if sensor_data:
    features = np.array([[
        sensor_data['ADS1115_0x48_A0'],
        sensor_data['ADS1115_0x48_A1'],
        sensor_data['ADS1115_0x48_A3'],
        sensor_data['ADS1115_0x49_A0'],
        sensor_data['ADS1115_0x49_A1']
    ]])
    prediction = clf.predict(features)
    print(f"Predicted Label: {prediction[0]}")
else:
    print("Failed to retrieve sensor data.")
