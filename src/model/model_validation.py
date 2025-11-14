import json
import shutil
import os

candidate_model = "models/candidate_model.pkl"
candidate_metrics = "results/candidate_metrics.json"
production_model = "models/production_model.pkl"
prod_metrics = "results/prod_metrics.json"
best_model = "models/best_model.pkl"
best_metrics = "results/best_metrics.json"

# Charger métriques modèle candidat
with open(candidate_metrics) as f:
    cand = json.load(f)
cand_score = cand.get("f1_score", 0.0)

# Charger métriques modèle production (si disponibles)
if os.path.exists(prod_metrics) and os.path.exists(production_model):
    with open(prod_metrics) as f:
        prod = json.load(f)
    prod_score = prod.get("f1_score", 0.0)
else:
    prod_score = 0.0  # aucun modèle en prod encore

print(f"Candidat: {cand_score:.3f}, Production: {prod_score:.4f}")

if not os.path.exists(production_model):
    # Première exécution → promotion directe
    print("⚠️ Aucun modèle de production trouvé, promotion directe du candidat")
    shutil.copy(candidate_model, best_model)
    shutil.copy(candidate_metrics, best_metrics)
elif cand_score > prod_score:
    print("✅ Nouveau modèle adopté")
    shutil.copy(candidate_model, best_model)
    shutil.copy(candidate_metrics, best_metrics)
else:
    print("⏭️ Ancien modèle conservé")
    shutil.copy(production_model, best_model)
    shutil.copy(prod_metrics, best_metrics)
