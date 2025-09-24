import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
import xgboost as xgb
import joblib

# === CONFIGURATION MLflow et AWS ===
from dotenv import load_dotenv
import os
load_dotenv()

# Ne mets plus de clés en dur ici

# Renseigne ces variables d’environnement si tu veux logguer sur AWS S3
bucket_name = "projetetss3"

# URI du serveur MLflow distant
mlflow.set_tracking_uri("http://ec2-35-183-108-241.ca-central-1.compute.amazonaws.com:5000/")
mlflow.set_experiment("HDFS")

# === CHEMINS DE FICHIERS ===
base_dir = "D:/ETSProject"
data_dir = os.path.join(base_dir, "data", "HDFS_results", "preprocessed")
model_dir = os.path.join(base_dir, "models")
os.makedirs(model_dir, exist_ok=True)

train_file = os.path.join(data_dir, "train_processed.csv")
valid_file = os.path.join(data_dir, "test_processed.csv")  # utilisé ici comme validation

# === CHARGEMENT DES DONNÉES ===
df_train = pd.read_csv(train_file)
df_valid = pd.read_csv(valid_file)

X_train = df_train.drop(columns=["BlockId", "Label"], errors='ignore')
y_train = df_train["Label"]

X_valid = df_valid.drop(columns=["BlockId", "Label"], errors='ignore')
y_valid = df_valid["Label"]

# === RANDOM FOREST ===
with mlflow.start_run(run_name="RandomForest-HDFS") as run:
    rf = RandomForestClassifier(n_estimators=10, random_state=42)

    X_train = X_train.select_dtypes(include=[np.number])
    X_valid = X_valid.select_dtypes(include=[np.number])

    rf.fit(X_train, y_train)

    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", 10)
    mlflow.log_param("random_state", 42)

    val_score = rf.score(X_valid, y_valid)
    mlflow.log_metric("val_accuracy", val_score)

    mlflow.sklearn.log_model(rf, "model")

    

    joblib.dump(rf, os.path.join(model_dir, "random_forest.pkl"))

    print(f"✅ RandomForest Accuracy: {val_score:.4f}")
    print(f"🔗 Run ID: {run.info.run_id}")

# === XGBOOST ===
with mlflow.start_run(run_name="XGBoost-HDFS") as run:
    model = xgb.XGBClassifier(n_estimators=100, max_depth=5, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    mlflow.log_param("model", "XGBoost")
    mlflow.log_param("n_estimators", 10)
    mlflow.log_param("max_depth", 5)

    val_score = model.score(X_valid, y_valid)
    mlflow.log_metric("val_accuracy", val_score)

    mlflow.sklearn.log_model(model, "model")
    joblib.dump(model, os.path.join(model_dir, "xgboost_model.pkl"))
    print(f"✅ XGBoost Accuracy: {val_score:.4f}")
    print(f"🔗 Run ID: {run.info.run_id}")
