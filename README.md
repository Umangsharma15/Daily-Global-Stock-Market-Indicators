
# 📈 End-to-End MLOps Project

## Daily Global Stock Market Prediction System

A **production-ready MLOps pipeline** that ingests stock data from MongoDB, performs automated validation, transformation, model training, evaluation, and deployment using **AWS, Docker, CI/CD, and FastAPI**.

This project demonstrates how to build a **scalable machine learning system** using real-world MLOps practices.

---

## 🚀 Project Highlights

* End-to-End automated ML pipeline
* MongoDB data ingestion
* Automated data validation & transformation
* Model versioning using AWS S3
* CI/CD with GitHub Actions
* Dockerized deployment
* FastAPI prediction service
* Deployed on AWS EC2

---

## 🧠 Problem Statement

Predict the **next closing price of a stock index** using global market indicators.

The system automatically:

1. Fetches data from MongoDB
2. Validates schema
3. Transforms features
4. Trains model
5. Evaluates against production model
6. Pushes best model to AWS S3
7. Deploys prediction API

---

## 🏗️ Project Architecture

```
MongoDB → Data Ingestion
        → Data Validation
        → Data Transformation
        → Model Trainer
        → Model Evaluation
        → Model Pusher (AWS S3)
        → FastAPI Prediction Service
        → Docker Container
        → AWS EC2 Deployment
```

---

## 🧰 Tech Stack

### 💻 Programming & ML

* Python 3.10
* Scikit-learn
* XGBoost
* Pandas
* NumPy

### 🗄️ Data & Storage

* MongoDB Atlas
* AWS S3 (Model registry)

### ⚙️ MLOps & DevOps

* Docker
* GitHub Actions (CI/CD)
* Self-Hosted Runner (EC2)

### 🌐 Backend & Deployment

* FastAPI
* Uvicorn
* AWS EC2

---

## 📁 Project Structure

```
src/
 ├── components/
 │   ├── data_ingestion.py
 │   ├── data_validation.py
 │   ├── data_transformation.py
 │   ├── model_trainer.py
 │   ├── model_evaluation.py
 │   └── model_pusher.py
 │
 ├── entity/
 │   ├── config_entity.py
 │   ├── artifact_entity.py
 │   └── s3_estimator.py
 │
 ├── pipeline/
 │   ├── training_pipeline.py
 │   └── prediction_pipeline.py
 │
 ├── configuration/
 │   ├── mongo_db_connection.py
 │   └── aws_connection.py
 │
 └── constants/
```

---

## 🔄 End-to-End Pipeline Flow

### 1️⃣ Data Ingestion

* Data fetched from **MongoDB Atlas**
* Converted into structured DataFrame
* Split into train/test

### 2️⃣ Data Validation

* Schema validation
* Column checks
* Data type verification

### 3️⃣ Data Transformation

* Feature engineering
* Scaling & encoding
* Pipeline object saved

### 4️⃣ Model Training

* Random Forest / XGBoost
* R² score evaluation
* Model artifact created

### 5️⃣ Model Evaluation

* Compare with production model
* Accept only if performance improves

### 6️⃣ Model Pusher

* Best model uploaded to **AWS S3**

---

## ☁️ AWS Services Used

| Service | Purpose              |
| ------- | -------------------- |
| AWS S3  | Model registry       |
| AWS EC2 | App deployment       |
| AWS ECR | Docker image storage |
| AWS IAM | Access control       |

---

## 🔁 CI/CD Pipeline

Implemented using **GitHub Actions**.

### Workflow:

1. Code pushed to GitHub
2. GitHub Action triggered
3. Docker image built
4. Image pushed to AWS ECR
5. EC2 self-hosted runner deploys container
6. App becomes live automatically

---

## 📊 Model Performance

| Metric   | Value |
| -------- | ----- |
| R² Score | ~0.58 |
| MAE      | ~0.95 |

---

## 📦 Key Features

* Modular ML pipeline
* Production model comparison
* Automatic model promotion
* Cloud model registry
* CI/CD automation
* Containerized deployment
* Real-time prediction API

---
💬 Connect
If you found this project helpful or have any questions, feel free to reach out!

This README provides a structured walkthrough of the MLOps project, showcasing the end-to-end pipeline, cloud integration, CI/CD setup, and robust data handling capabilities.





