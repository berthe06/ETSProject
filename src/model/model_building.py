import os
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
import xgboost as xgb
import joblib
from dotenv import load_dotenv
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,  # pour AUPRC
    accuracy_score,          # si tu l’utilises ailleurs
)


# === CONFIGURATION MLflow et AWS ===
load_dotenv()
mlflow.set_tracking_uri("http://ec2-35-182-211-61.ca-central-1.compute.amazonaws.com:5000/")
mlflow.set_experiment("HDFS")

# === CHEMINS ===
base_dir = "D:/ETSProject"
data_dir = os.path.join(base_dir, "data", "HDFS_results", "preprocessed")
model_dir = os.path.join(base_dir, "models")
results_dir = os.path.join(base_dir, "results")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

train_file = os.path.join(data_dir, "train_processed.csv")
valid_file = os.path.join(data_dir, "test_processed.csv")

# === CHARGEMENT DES DONNÉES ===
df_train = pd.read_csv(train_file)
df_valid = pd.read_csv(valid_file)

X_train = df_train.drop(columns=["BlockId", "Label"], errors="ignore")
y_train = df_train["Label"]
X_valid = df_valid.drop(columns=["BlockId", "Label"], errors="ignore")
y_valid = df_valid["Label"]

# Garder uniquement les colonnes numériques
X_train = X_train.select_dtypes(include=[np.number])
X_valid = X_valid.select_dtypes(include=[np.number])

# Initialisation
rf_score, xgb_score = None, None
rf_path, xgb_path = None, None

# === RANDOM FOREST ===
with mlflow.start_run(run_name="RandomForest-HDFS") as run:
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    # Prédictions
    y_pred  = rf.predict(X_valid)
    y_proba = rf.predict_proba(X_valid)[:, 1]  # pour AUPRC

    # Métriques
    acc   = (y_pred == y_valid).mean()
    prec  = precision_score(y_valid, y_pred, zero_division=0)
    rec   = recall_score(y_valid, y_pred, zero_division=0)
    f1    = f1_score(y_valid, y_pred, zero_division=0)
    auprc = average_precision_score(y_valid, y_proba)

    # Log MLflow
    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("val_accuracy", acc)
    mlflow.log_metric("val_precision", prec)
    mlflow.log_metric("val_recall", rec)
    mlflow.log_metric("val_f1", f1)
    mlflow.log_metric("val_auprc", auprc)

    # (optionnel) log du modèle
    mlflow.sklearn.log_model(rf, "model")

    # Sauvegarde locale
    rf_path = os.path.join(model_dir, "random_forest.pkl")
    joblib.dump(rf, rf_path)

    print(f"✅ RF | Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}  AUPRC={auprc:.4f}")
    print(f"🔗 Run ID: {run.info.run_id}")


# === XGBOOST ===
with mlflow.start_run(run_name="XGBoost-HDFS") as run:
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        eval_metric="logloss",   # on calcule AUPRC nous-mêmes ci-dessous
        use_label_encoder=False,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)

    # Prédictions
    y_pred  = xgb_model.predict(X_valid)
    y_proba = xgb_model.predict_proba(X_valid)[:, 1]  # pour AUPRC

    # Métriques
    acc   = (y_pred == y_valid).mean()
    prec  = precision_score(y_valid, y_pred, zero_division=0)
    rec   = recall_score(y_valid, y_pred, zero_division=0)
    f1    = f1_score(y_valid, y_pred, zero_division=0)
    auprc = average_precision_score(y_valid, y_proba)

    # Log MLflow
    mlflow.log_param("model", "XGBoost")
    mlflow.log_param("n_estimators", 20)
    mlflow.log_param("max_depth", 5)
    mlflow.log_metric("val_accuracy", acc)
    mlflow.log_metric("val_precision", prec)
    mlflow.log_metric("val_recall", rec)
    mlflow.log_metric("val_f1", f1)
    mlflow.log_metric("val_auprc", auprc)

    # (optionnel) log du modèle — tu peux commenter pendant le tuning pour gagner du temps réseau
    mlflow.xgboost.log_model(xgb_model, "model")

    # Sauvegarde locale
    xgb_path = os.path.join(model_dir, "xgboost_model.pkl")
    joblib.dump(xgb_model, xgb_path)

    print(f"✅ XGBoost | Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}  AUPRC={auprc:.4f}")
    print(f"🔗 Run ID: {run.info.run_id}")


# === COMPARAISON ===

rf_score = f1_score(y_valid, rf.predict(X_valid), zero_division=0)
xgb_score = f1_score(y_valid, xgb_model.predict(X_valid), zero_division=0)


if xgb_score > rf_score:
    best_name, best_score, best_path = "XGBoost", xgb_score, xgb_path
else:
    best_name, best_score, best_path = "RandomForest", rf_score, rf_path

candidate_model = os.path.join(model_dir, "candidate_model.pkl")
joblib.dump(joblib.load(best_path), candidate_model)

metrics = {"model": best_name, "val_accuracy": best_score}
with open(os.path.join(results_dir, "candidate_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

print(f"🏆 Meilleur modèle : {best_name} avec Accuracy = {best_score:.4f}")
print(f"📦 Sauvé comme {candidate_model} (copie de {best_path})")
