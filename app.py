import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)
# Load the model
model = pickle.load(open('vehicle_pred_model1.pkl', 'rb'))

def convert_to_native_types(data):
    converted_data = {}
    for key, value in data.items():
        try:
            # Attempt to convert the value to a native Python type
            converted_data[key] = np.asscalar(np.float64(value)) if isinstance(value, np.float32) else value
        except Exception as e:
            converted_data[key] = value
    return converted_data

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    # Access form data using request.form instead of request.json
    data = request.form.to_dict()
    data = convert_to_native_types(data)

    # Adjust column names here according to your model
    column_names = ['year', 'fuelConsump_per_100km', 'kilometers', 'brand', 'transmission', 'drive', 'fuel', 'body_style', 'seats', 'engine']

    # Create a dictionary with default values for all columns
    default_data = {col: None for col in column_names}

    # Update default_data with received data
    default_data.update(data)

    # Example: Adjusting input data to match the expected 10 features
    new_data = np.array(list(default_data.values()))[:10].reshape(1, -1)

    output = model.predict(new_data)
    return render_template('home.html', prediction_text=f"Predicted output: {output[0]}")

if __name__ == "__main__":
    app.run(debug=True)
