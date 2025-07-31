import os
from dotenv import load_dotenv
import os
load_dotenv()

# Ne mets plus de clés en dur ici

# === Répertoire racine du projet
BASE_DIR = "D:/ETSProject"

# === Répertoires pour les données
DATA_DIR = os.path.join(BASE_DIR, "data", "HDFS_results", "preprocessed")
RAW_LOG_DIR = os.path.join(BASE_DIR, "data", "HDFS_raw_logs")

# Fichiers de données
TRAIN_FILE = os.path.join(DATA_DIR, "train_processed.csv")
VALID_FILE = os.path.join(DATA_DIR, "test_processed.csv")
TEST_FILE  = os.path.join(DATA_DIR, "test_processed.csv")  # si tu en sépares un jour

# === Répertoire pour les modèles sauvegardés
MODEL_DIR = os.path.join(BASE_DIR, "models")
RANDOM_FOREST_PATH = os.path.join(MODEL_DIR, "random_forest.pkl")
XGBOOST_MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_model.pkl")

# === Configuration AWS S3

S3_BUCKET_NAME = "projetetss3"

# === Configuration MLflow
MLFLOW_TRACKING_URI = "http://ec2-18-207-206-140.compute-1.amazonaws.com:5000"
EXPERIMENT_NAME = "HDFS"
