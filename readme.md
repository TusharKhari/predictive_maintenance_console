# AI4I Predictive Maintenance System

An end-to-end, production-ready machine learning system that forecasts equipment failure from real-time operating telemetry.

https://predictivemaintenanceconsole-sgmntdcathmhurbtet2u4s.streamlit.app/

---

## 📌 Problem & Context

Unexpected equipment breakdown causes significant production downtime and maintenance costs. The goal of this system is to identify early failure signals from machine sensor streams, calculate failure probabilities, extract the top operational risk drivers via SHAP, and surface actionable alerts to operators before critical breakdown occurs.

> **Dataset Notice:** Built using the [UCI AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) (10,000 synthetic records reflecting real-world industrial milling machine telemetry).

---

## 🏗️ System Architecture

```text
                     INTERNET / CLIENT
                            │
                            ▼
                  ┌───────────────────┐
                  │   Vercel / Web    │
                  │   React Dashboard │
                  │  (Tailwind, Vite) │
                  └─────────┬─────────┘
                            │ HTTPS / JSON
                            ▼
                  ┌───────────────────┐
                  │   Render / Host   │
                  │  FastAPI Backend  │
                  │   (Pydantic v2)   │
                  └─────┬───────┬─────┘
                        │       │
       Inference Engine │       │ Audit Logging
                        ▼       ▼
    ┌──────────────────────┐  ┌─────────────┐
    │ ML Pipeline Artifact │  │ PostgreSQL  │
    │ Encoders + Scalers + │  │ Prediction  │
    │ XGBoost + SHAP Tree  │  │ History Log │
    └──────────────────────┘  └─────────────┘


```
## 🛠️ Technology Stack

| Layer | Technologies |
|---|---|
| **ML & Analytics** | Python, Scikit-learn, XGBoost, Pandas, NumPy |
| **Backend API** | FastAPI, Uvicorn|
| **Frontend UI** | Streamlit |
<!-- | **Database** | PostgreSQL |
| **Container & CI/CD** | Docker, Docker Compose, GitHub Actions |
| **Cloud Hosting** | Render (API container), Vercel (Frontend SPA) | -->


## Build/Trained Different Models

1. Logistic Regression
2. Random Forest
3. XGBoost

## Model Evaluation
The models will be compared using several classification metrics:

- Accuracy - Precision - Recall - F1 Score - ROC-AUC - PR-AUC - Confusion Matrix

# Model Performance

The following table summarizes the performance of the three classification models evaluated on the test set.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9675 | 0.5714 | 0.1765 | 0.2697 | 0.9257 | 0.4622 |
| Random Forest | **0.9910** | **0.9464** | **0.7794** | **0.8548** | 0.9642 | 0.8769 |
| XGBoost | **0.9910** | **0.9464** | **0.7794** | **0.8548** | **0.9787** | **0.8835** |






