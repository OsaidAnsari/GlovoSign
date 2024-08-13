from flask import Flask, jsonify
from flask_cors import CORS
import requests
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})  # Corrected URL format

model_filename = 'cnn_model.h5'
try:
    model = load_model(model_filename)  # Load the Keras model
except FileNotFoundError:
    print(f"Model file {model_filename} not found.")
    model = None

nodemcu_url = "http://192.168.41.19:100/"  # Replace with the actual URL

# Define the mapping from numerical labels to string labels
label_mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}  # Modify according to your specific labels

def get_sensor_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

@app.route('/conversations/<conversation_id>', methods=['GET'])
def get_prediction(conversation_id):
    if model is None:
        return jsonify({'error': 'Model not loaded.'}), 500
    
    sensor_data = get_sensor_data(nodemcu_url)
    if sensor_data:
        try:
            features = np.array([[
                sensor_data.get('ADS1115_0x48_A0', 0),
                sensor_data.get('ADS1115_0x48_A1', 0),
                sensor_data.get('ADS1115_0x48_A3', 0),
                sensor_data.get('ADS1115_0x49_A0', 0),
                sensor_data.get('ADS1115_0x49_A1', 0)
            ]])
            
            # Expand dimensions for CNN (assuming 1D Conv1D)
            features = np.expand_dims(features, axis=2)
            
            # Make prediction
            prediction = model.predict(features)
            predicted_class = np.argmax(prediction[0])  # Get the predicted class index
            
            # Convert numerical prediction to string label
            predicted_label = label_mapping.get(predicted_class, "Unknown Label")
            
            return jsonify({'prediction': predicted_label})
        except KeyError as e:
            print(f"Key error: {e}")
            return jsonify({'error': 'Invalid sensor data format.'}), 500
    else:
        return jsonify({'error': 'Failed to retrieve sensor data.'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
