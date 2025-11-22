import os
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
import json

# Charger les variables d'environnement
load_dotenv()

# Configuration de MLflow
mlflow.set_tracking_uri("http://ec2-35-182-211-61.ca-central-1.compute.amazonaws.com:5000/")
mlflow.set_experiment("HDFS")

# =========================
# Répertoires
# =========================
base_dir = "D:/ETSProject"
data_dir = os.path.join(base_dir, "data", "HDFS_results", "preprocessed")
model_dir = os.path.join(base_dir, "models")

test_file = os.path.join(data_dir, "test_processed.csv")
model_path = os.path.join(model_dir, "random_forest.pkl")

# =========================
# Chargement des données de test
# =========================
df_test = pd.read_csv(test_file)

X_test = df_test.drop(columns=["BlockId", "Label"], errors="ignore")
y_test = df_test["Label"]

# 🔧 Assure-toi d’avoir les bonnes features
X_test = X_test.select_dtypes(include=[np.number])  # ← essentiel !

# =========================
# Chargement du modèle
# =========================
model = joblib.load(model_path)

# =========================
# Évaluation
# =========================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("📊 Résultats de l’évaluation :")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

print("\n📄 Rapport de classification :")
print(classification_report(y_test, y_pred))

# =========================
# Log des résultats dans MLflow
# =========================
with mlflow.start_run(run_name="Evaluation-RandomForest"):
    mlflow.log_param("model", "RandomForest")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(model, artifact_path="model_rf_eval")

# =========================
# Sauvegarde des métriques localement
# =========================

# Créer le dossier results/ s’il n’existe pas
results_dir = os.path.join(base_dir, "results")
os.makedirs(results_dir, exist_ok=True)

# Fichier de sortie attendu par DVC
metrics_path = os.path.join(results_dir, "evaluation_metrics.json")

# Écriture des métriques dans le fichier JSON
metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1
}

with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)

print(f"\n✅ Métriques sauvegardées dans : {metrics_path}")
