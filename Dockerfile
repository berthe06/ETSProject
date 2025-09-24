FROM python:3.11-slim

# Set working directory
WORKDIR /app

# ✅ Installe git et autres dépendances nécessaires
RUN apt-get update && \
    apt-get install -y --no-install-recommends git gcc && \
    rm -rf /var/lib/apt/lists/*

# Copie les dépendances Python
COPY requirements.txt .

# Installe les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copie le reste du projet
COPY . .

# Commande par défaut (modifie si besoin)
CMD ["python", "src/model/model_registration.py"]
