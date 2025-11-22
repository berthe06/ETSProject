# ETSProject

# url ecr = 597571726194.dkr.ecr.ca-central-1.amazonaws.com/mlproject


# HDFS Log Anomaly Detection – MLOps Pipeline

This repository contains the code and MLOps pipeline developed as part of my master’s thesis at ÉTS.  
The goal is to **detect anomalies in HDFS logs** using a **Random Forest** model, with a fully automated and reproducible workflow:

- Data versioning and pipelines with **DVC**
- Experiment tracking with **MLflow**
- Containerization with **Docker**
- Continuous Integration & Delivery with **GitHub Actions**
- Deployment on **AWS (ECR + EC2, self-hosted runner)**

---

## 1. Project Overview

Modern distributed systems (like Hadoop clusters) generate huge amounts of logs.  
Manually detecting anomalies is:

- time-consuming  
- error-prone  
- difficult to scale  

This project builds an **end-to-end MLOps solution** that:

1. Ingests and preprocesses HDFS log data  
2. Trains and evaluates a Random Forest anomaly detection model  
3. Tracks experiments and artefacts  
4. Builds and pushes a Docker image on each change in `main`  
5. Automatically deploys the latest model as a REST API on an EC2 instance

---

## 2. Architecture

**Main components:**

- `DVC` – data and pipeline orchestration (`dvc.yaml`, `params.yaml`)
- `MLflow` – experiment tracking (metrics, params, models)
- `Docker` – containerized inference service (Flask/FastAPI app)
- `GitHub Actions` – CI/CD pipelines
- `AWS ECR` – Docker image registry
- `AWS EC2` – self-hosted GitHub runner and deployment target

### High-level CI/CD flow:

1. Developer pushes to `main`
2. GitHub Actions runs CI checks + builds Docker image
3. Image is pushed to **ECR**
4. A self-hosted runner on **EC2** pulls the new image
5. Old container is replaced by the new one
6. The API is exposed on port `8080`

---

## 3. Repository Structure

```text
.
├── data/                     
├── src/
│   ├── data_ingestion/       
│   ├── data_preprocessing/   
│   ├── model/                
│   └── app/                  
├── dvc.yaml                  
├── params.yaml               
├── Dockerfile                
├── requirements.txt          
├── .github/workflows/
│   └── cicd.yml              
└── README.md
```

---

## 4. Local Setup

### Requirements

- Python 3.10+
- Git
- DVC
- Docker

### Installation

```bash
git clone https://github.com/berthe06/ETSProject.git
cd ETSProject
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run Pipeline

```bash
dvc pull
dvc repro
```

---

## 5. MLflow Tracking

```bash
mlflow ui --backend-store-uri mlruns
```

Runs at `http://127.0.0.1:5000`

---

## 6. Docker & API

### Build and run

```bash
docker build -t hdfs-anomaly-api .
docker run -d -p 8080:8080 --name=hdfs-anomaly-api hdfs-anomaly-api
```

API runs at:

```
http://localhost:8080
```

---

## 7. CI/CD with GitHub Actions & AWS

### CI

- Runs on push to `main`
- Builds and pushes Docker image to AWS ECR

### CD

- Runs on EC2 via self-hosted runner  
- Pulls latest image  
- Removes old container  
- Starts new one  
- Zero-touch deployment  

---

## 8. GitHub Secrets Required

```
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_REGION
AWS_ECR_LOGIN_URI
ECR_REPOSITORY_NAME
```

---

## 9. Future Work

- Drift detection  
- Automated retraining  
- MLflow-based model selection  
- Monitoring (Prometheus/CloudWatch)  
- Blue/Green deployment  

---

## 10. Author

**Lema Noah Bernadette**  
Master’s student – ÉTS  
HDFS Log Anomaly Detection – MLOps Pipeline
