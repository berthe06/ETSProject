import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import mlflow
import mlflow.sklearn
import numpy as np


from dotenv import load_dotenv
import os
load_dotenv()


mlflow.set_tracking_uri("http://ec2-18-207-206-140.compute-1.amazonaws.com:5000")
mlflow.set_experiment("HDFS")

# Répertoires
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

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)

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
