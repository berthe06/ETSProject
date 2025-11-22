import os
import mlflow
import joblib
from dotenv import load_dotenv

# Charger les variables d’environnement (MLFLOW_TRACKING_URI, etc.)
load_dotenv()

# === CONFIGURATION MLflow ===
mlflow.set_tracking_uri("http://ec2-35-182-211-61.ca-central-1.compute.amazonaws.com:5000/")
mlflow.set_experiment("HDFS")

# === CHEMINS ===
model_dir = "D:/ETSProject/models"
model_path = os.path.join(model_dir, "random_forest.pkl")

# === PARAMÈTRES DE REGISTRATION ===
registered_model_name = "RandomForest_HDFS"
run_name = "Register-RandomForest"

# === Chargement du modèle local ===
model = joblib.load(model_path)

# === Lancement du run MLflow ===
with mlflow.start_run(run_name=run_name) as run:
    # Log du modèle local
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model_rf_registered",
        registered_model_name=registered_model_name
    )

    print("✅ Modèle enregistré dans le registry MLflow avec le nom :", registered_model_name)
    print("🔗 Run ID:", run.info.run_id)

# Créer le dossier de registre local (pour DVC)
os.makedirs("model_registry", exist_ok=True)

# Sauvegarder une copie locale du modèle
joblib.dump(model, "model_registry/random_forest.pkl")
