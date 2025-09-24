# Utilise une image officielle Python
FROM python:3.11-slim

# Crée un répertoire de travail
WORKDIR /app

# Copie les fichiers de dépendances
COPY requirements.txt .

# Installe les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copie tout le projet (sources, données, etc.)
COPY . .

# (Optionnel) Définit une commande de base si tu veux que ça lance une étape DVC automatiquement
# CMD ["dvc", "repro"]
