# 🌱 Greenwashing Detection MLOps Pipeline using Apache Airflow, DagsHub & Drift Monitoring

## 📌 Business Problem

Many organizations publicly claim to be environmentally responsible while their reported ESG indicators suggest otherwise. This practice, known as **Greenwashing**, makes it difficult for investors, regulators, and stakeholders to identify genuinely sustainable companies.

This project builds an **end-to-end MLOps pipeline** to detect potential greenwashing using ESG data while demonstrating production-oriented workflow orchestration, experiment tracking, artifact management, continuous integration, and model monitoring.

---

## 🎥 Demo

**Project Demo**

https://vimeo.com/1204773035?share=copy&fl=sv&fe=ci

---

# Architecture

```text
                       ESG CSV Dataset
                              │
                              ▼
                 Airflow ETL Pipeline DAG
         (Extract → Transform → Validate → Load)
                              │
                              ▼
                        PostgreSQL
                    (Central Data Store)
                              │
                              ▼
              Airflow Training Pipeline DAG
      Data Processing
      → Train/Test Split
      → Feature Engineering
      → Model Training
      → Model Evaluation
                              │
             ┌────────────────┴────────────────┐
             │                                 │
             ▼                                 ▼
      Trained Model                     DagsHub (MLflow)
                                        • Experiment Tracking
                                        • Dataset Versioning
                                        • Model Artifacts
                                        • Processed Artifacts
                              │
                              ▼
              Airflow Monitoring Pipeline DAG
      • Evidently Data Drift Detection
      • Model Performance Monitoring (MODEL DRIFT)
      • Drift Metrics Logging
                              │
             ┌────────────────┴────────────────┐
             │                                 │
             ▼                                 ▼
      PostgreSQL                     DagsHub
      Prediction Logs                Monitoring Metrics
                                     Drift Reports
```

---

# Tech Stack

* Apache Airflow (**workflow orchestrator**)
* PostgreSQL  (**datawarehouse**)
* Scikit-learn
* Evidently AI
* DagsHub (MLflow)
* GitHub Actions
* Docker
* SQLAlchemy
* Pandas
* Joblib

---

**Docker**: Containerized Apache Airflow using Docker to orchestrate ETL, training, and monitoring workflows in a reproducible environment.

---

### Data Drift

Implemented using **Evidently AI**.

Detects changes in incoming ESG data distribution compared to the training data.

### Model Monitoring

Tracks:

* Accuracy (0.98)
* Precision (1.0)
* Recall (0.92)
* F1 Score (0.96)

Performance degradation is logged over time.

* Accuracy drop
* Precision drop
* Recall drop
* F1 drop
* Drift Score
* Drift Detected

Monitoring metrics are:

* *Stored in PostgreSQL*
* *Logged to DagsHub*

---

# Simulating Production Data

Since this project is not connected to a live prediction API, there are no real incoming prediction requests.

To simulate a production environment:

* 2024 ESG records were used to generate synthetic 2025 and 2026 datasets.
* These synthetic records represent "current production data".

This mimics a real deployment where:

Reference Data = Training Dataset

Current Data = Incoming Prediction Requests

allowing realistic demonstration of drift detection.

---

# CI Pipeline

GitHub Actions is used for Continuous Integration.

Current CI performs:

* Dependency installation
* Code formatting verification using Black

This ensures every push is automatically validated before merging.

Airflow orchestration remains independent of CI.

---

# DagsHub Integration

The project uses DagsHub as the MLflow Tracking Server for:

* Experiment Tracking (metrics, hyper parameters)
* Dataset Versioning (train)
* Model Artifact Storage (Random Forest)
* Processed Artifact Storage (OneHotEncoded, PowerTransformer)
* Monitoring Metric Logging 

*Training and monitoring pipelines log directly to DagsHub from within Airflow tasks.*

---
# ML 
**Power Transformer** for skewed feature normalization, **SMOTE** for class imbalance handling, **feature engineering**, train/test splitting, Random Forest model training.

---
*After starting Astro (e.g., **astro dev start**), DAGs are **triggered** from the Airflow UI, where task execution, logs, retries, and success/failure status can be monitored in real time.*
