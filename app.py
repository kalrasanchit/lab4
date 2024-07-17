from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and scaler
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [
        float(request.form['length1']),
        float(request.form['length2']),
        float(request.form['length3']),
        float(request.form['height']),
        float(request.form['width'])
    ]

    # Scale the features
    scaled_features = scaler.transform(np.array(features).reshape(1, -1))

    # Make prediction
    prediction = model.predict(scaled_features)

    return jsonify({'species': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
