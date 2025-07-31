from flask import Flask, request, jsonify
from model_loader import load_model, predict_log
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
model = load_model()

@app.route('/')
def home():
    return "API de prédiction d'anomalies dans les logs HDFS"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    log_line = data.get('log', '')
    
    if not log_line:
        return jsonify({'error': 'log non fourni'}), 400

    prediction = predict_log(model, log_line)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
