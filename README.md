<div align="center">

# ❤️ CardioAI (HeartCare-AI) — Clinical Cardiovascular Risk Prediction System

**An End-to-End Machine Learning Healthcare Platform Featuring Dual Interfaces (FastAPI + Glassmorphism Web Portal & Streamlit Dashboard) and Automated Clinical Report Exports.**

[![FastAPI](https://img.shields.io/badge/FastAPI-0.100.0+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.6.1-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-3.10--slim-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Bootstrap 5](https://img.shields.io/badge/Bootstrap-5.3.0-7952B3?style=for-the-badge&logo=bootstrap&logoColor=white)](https://getbootstrap.com/)
[![Live Demo](https://img.shields.io/badge/Live_Demo-Render-46E3B7?style=for-the-badge&logo=render&logoColor=black)](https://heartcare-ai-jwy4.onrender.com)

[Live Demo](https://heartcare-ai-jwy4.onrender.com) • [Key Features](#-key-features) • [ML Pipeline & Schema](#-machine-learning-pipeline--clinical-features) • [Installation & Docker](#-getting-started--deployment) • [API Reference](#-api-reference)

</div>

---

## 📖 Overview

**CardioAI (HeartCare-AI)** is a predictive medical analytics web application engineered to evaluate a patient's probability of cardiovascular disease based on 11 clinical, demographic, and electrocardiographic (ECG) parameters[cite: 35, 36, 45, 47].

The system is powered by a pre-trained **Scikit-Learn K-Nearest Neighbors (`KNeighborsClassifier`)** model paired with a **StandardScaler** and dynamic 15-feature one-hot schema alignment[cite: 41, 42, 43, 47, 49]. It supports two deployment modes:
1. **Production Web Portal (`main.py` + `healthcare_AI.html`):** A **FastAPI** microservice serving a modern Bootstrap 5 Glassmorphism UI with client-side **PDF and PNG clinical diagnostic report generation** (`html2canvas` + `jsPDF`).
2. **Interactive Data Science Dashboard (`app.py`):** A standalone **Streamlit** application with batched form sliders and real-time risk probability gauges.

---

## ✨ Key Features

### 🧠 Machine Learning & Backend Architecture
* **Serialized Model & Preprocessing Artifacts:** Uses `joblib` to load `best_model.pkl` (`KNeighborsClassifier` with KD-Tree & Euclidean/Minkowski distance), `scaler.pkl` (`StandardScaler`), and `columns.pkl`[cite: 41, 42, 47, 49].
* **Dynamic One-Hot Schema Alignment:** Automatically converts raw categorical inputs (`Sex`, `ChestPainType`, `RestingECG`, `ExerciseAngina`, `ST_Slope`) via `pd.get_dummies()` and enforces exact 15-column training alignment using `.reindex(columns=columns, fill_value=0)` before scaling[cite: 41, 43, 47].
* **Probabilistic Risk Scoring:** Computes both binary classification (`0` = Low Risk, `1` = High Risk) and exact disease risk probability percentage via `model.predict_proba()`[cite: 41, 45, 47].
* **Containerized & CORS-Enabled:** Includes a production-ready `Dockerfile` (`python:3.10-slim`) and full `CORSMiddleware` support for decoupled frontend-backend hosting.

### 🩺 Clinical Web Portal & Reporting (`healthcare_AI.html`)
* **Structured Clinical Input Sections:** Organizes 11 patient metrics into three intuitive medical groups: *Demographics & Symptoms*, *Clinical Vitals*, and *ECG & Exercise Metrics*.
* **Animated Risk Assessment Dashboard:** Features smooth count-up probability animations, color-coded risk badges (`HIGH RISK` / `LOW RISK`), and cubic-bezier progress bars[cite: 45].
* **Instant Medical Report Export (PDF & PNG):** Renders an off-screen formal clinical report template complete with a unique Report ID (`C-AI-XXXXXX`), timestamp, patient demographics, full 9-row clinical vitals table, AI risk score, and medical recommendations—exportable in one click as a high-resolution **PNG** or **A4 PDF** directly on the client device[cite: 45].

---

## 🔬 Machine Learning Pipeline & Clinical Features

### Input Clinical Parameters (11 Raw Inputs)

| Parameter | Field Name | Type / Range | Clinical Description |
| :--- | :--- | :--- | :--- |
| **Age** | `Age` | Integer (`18 – 120`) | Patient's age in years[cite: 41, 45, 47] |
| **Sex** | `Sex` | `Male` (`M`) / `Female` (`F`) | Biological sex of the patient[cite: 41, 45, 47] |
| **Chest Pain Type** | `ChestPainType` | `ATA`, `NAP`, `ASY`, `TA` | Atypical Angina, Non-Anginal Pain, Asymptomatic, or Typical Angina[cite: 41, 45, 47] |
| **Resting BP** | `RestingBP` | Integer (`80 – 220` mmHg) | Resting blood pressure in mm Hg[cite: 41, 45, 47] |
| **Cholesterol** | `Cholesterol` | Integer (`0 – 600` mg/dl) | Serum cholesterol level in mg/dl[cite: 41, 45, 47] |
| **Fasting Blood Sugar** | `FastingBS` | `0` (`<= 120`) / `1` (`> 120`) | Fasting blood sugar > 120 mg/dl[cite: 41, 45, 47] |
| **Resting ECG** | `RestingECG` | `Normal`, `ST`, `LVH` | Normal, ST-T wave abnormality, or Left ventricular hypertrophy[cite: 41, 45, 47] |
| **Max Heart Rate** | `MaxHR` | Integer (`60 – 220` bpm) | Maximum heart rate achieved during exercise[cite: 41, 45, 47] |
| **Exercise Angina** | `ExerciseAngina` | `No` (`N`) / `Yes` (`Y`) | Exercise-induced angina[cite: 41, 45, 47] |
| **Oldpeak** | `Oldpeak` | Float (`-2.0 – 10.0`) | ST depression induced by exercise relative to rest[cite: 41, 45, 47] |
| **ST Slope** | `ST_Slope` | `Up`, `Flat`, `Down` | Slope of the peak exercise ST segment[cite: 41, 45, 47] |

### Aligned 15-Feature Vector (`columns.pkl`)
During inference, the 11 raw inputs are transformed and aligned into the following exact 15-feature vector before `StandardScaler` normalization[cite: 41, 43, 47, 49]:
```python
[
  "Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak",
  "Sex_M", "ChestPainType_ATA", "ChestPainType_NAP", "ChestPainType_TA",
  "RestingECG_Normal", "RestingECG_ST", "ExerciseAngina_Y",
  "ST_Slope_Flat", "ST_Slope_Up"
]
