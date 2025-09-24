FROM python:3.11-slim

# Répertoire de travail
WORKDIR /app

# Installer git et dépendances nécessaires à la compilation des wheels
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    build-essential \
    gcc \
    python3-dev \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# Copier requirements.txt et installer dépendances Python
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste du projet
COPY . .

# Commande par défaut (adapter selon ton besoin)
CMD ["python", "src/model/model_registration.py"]
