import joblib
import os

def load_model():
    model_path = os.getenv('MODEL_PATH', 'models/random_forest.pkl')
    return joblib.load(model_path)

def predict_log(model, log_line):
    features = extract_features_from_log(log_line)
    return int(model.predict([features])[0])

def extract_features_from_log(log_line):
    # À personnaliser selon ta méthode de vectorisation
    return [0] * 50  # Valeur fictive
