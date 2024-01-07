from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from joblib import load

app = Flask(__name__)

# Load the saved components
knn_model = load('./Data/knn_model.joblib')


@app.route('/')
def home():
    return render_template('./templates/index.html')  # Simple HTML form for file upload

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        data = pd.read_csv(file)

        # Making predictions
        predictions = knn_model.predict(data)
        
        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})



if __name__ == '__main__':

    print("hiii")
    app.run(debug=True)
