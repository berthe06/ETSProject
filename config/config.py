# config/config.py

MODEL_NAME = "random_forest"
EXPERIMENT_NAME = "HDFS"
BUCKET_NAME = "mlops-bucket-25"

TRAIN_DATA_PATH = "data/HDFS_results/preprocessed/train_processed.csv"
TEST_DATA_PATH = "data/HDFS_results/preprocessed/test_processed.csv"

RANDOM_SEED = 42

# === MLflow Tracking URI
MLFLOW_TRACKING_URI = "http://ec2-3-96-152-110.ca-central-1.compute.amazonaws.com:5000/" # distant
# MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"  # si tu veux localement
