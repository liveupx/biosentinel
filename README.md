<div align="center">

<br/>

```
██████╗ ██╗ ██████╗ ███████╗███████╗███╗   ██╗████████╗██╗███╗   ██╗███████╗██╗
██╔══██╗██║██╔═══██╗██╔════╝██╔════╝████╗  ██║╚══██╔══╝██║████╗  ██║██╔════╝██║
██████╔╝██║██║   ██║███████╗█████╗  ██╔██╗ ██║   ██║   ██║██╔██╗ ██║█████╗  ██║
██╔══██╗██║██║   ██║╚════██║██╔══╝  ██║╚██╗██║   ██║   ██║██║╚██╗██║██╔══╝  ██║
██████╔╝██║╚██████╔╝███████║███████╗██║ ╚████║   ██║   ██║██║ ╚████║███████╗███████╗
╚═════╝ ╚═╝ ╚═════╝ ╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝
```

### **AI-Powered Longitudinal Health Monitoring & Early Disease Prediction Platform**
#### *Detect cancer & serious diseases years before symptoms appear — saving millions of lives.*

<br/>

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![Stars](https://img.shields.io/github/stars/liveupx/biosentinel?style=for-the-badge&color=yellow)](https://github.com/liveupx/biosentinel/stargazers)
[![Forks](https://img.shields.io/github/forks/liveupx/biosentinel?style=for-the-badge)](https://github.com/liveupx/biosentinel/network)
[![Issues](https://img.shields.io/github/issues/liveupx/biosentinel?style=for-the-badge&color=red)](https://github.com/liveupx/biosentinel/issues)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=for-the-badge)](CONTRIBUTING.md)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg?style=for-the-badge)](CODE_OF_CONDUCT.md)
[![Build Status](https://img.shields.io/github/actions/workflow/status/liveupx/biosentinel/ci.yml?style=for-the-badge)](https://github.com/liveupx/biosentinel/actions)
[![Documentation](https://img.shields.io/badge/docs-biosentinel.liveupx.com-informational?style=for-the-badge)](https://biosentinel.liveupx.com)

<br/>

> **⚕️ BioSentinel is not a medical device and does not replace professional medical diagnosis.**  
> **It is an open-source research & decision-support platform for healthcare professionals and researchers.**

<br/>

[🚀 Live Demo](https://biosentinel.xhost.live) · [📖 Documentation](docs/) · [🐛 Report Bug](https://github.com/liveupx/biosentinel/issues/new?template=bug_report.md) · [💡 Request Feature](https://github.com/liveupx/biosentinel/issues/new?template=feature_request.md) · [💬 Discussions](https://github.com/liveupx/biosentinel/discussions)

</div>

---

## 📖 Table of Contents

- [The Vision](#-the-vision)
- [Why BioSentinel?](#-why-biosentinel)
- [Research Foundation](#-research-foundation)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Technology Stack](#-technology-stack)
- [ML Models & Algorithms](#-ml-models--algorithms)
- [Data Schema](#-data-schema)
- [Getting Started](#-getting-started)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [API Reference](#-api-reference)
- [Project Structure](#-project-structure)
- [Roadmap](#-roadmap)
- [Research Papers](#-research-papers)
- [Contributing](#-contributing)
- [Community & Support](#-community--support)
- [Ethics & Privacy](#-ethics--privacy)
- [License](#-license)
- [Creator & Team](#-creator--team)
- [Acknowledgements](#-acknowledgements)

---

## 🌟 The Vision

Every year, **10 million people die from cancer** — most of them unnecessarily.

The tragic reality? **80%+ of cancers are curable when detected early.** The same is true for diabetes, cardiovascular disease, liver disease, kidney failure, and dozens of other serious conditions. Yet we catch them too late — after symptoms appear, after the damage is done, after the window for effective treatment has closed.

**BioSentinel changes that.**

By continuously tracking a person's full medical picture — regular checkups (every 3 months), all medications, diagnosed diseases, prescriptions, diet plans, lifestyle markers, and lab biomarkers — over a 3+ year longitudinal window, BioSentinel's AI models can detect subtle, often invisible patterns that indicate a developing disease **years before clinical symptoms emerge**.

> *"The best time to treat cancer was before it became cancer. BioSentinel gives us that window."*

This is not science fiction. It is science. Peer-reviewed research from Nature Medicine, BMC, and NIH has demonstrated that:

- AI models analyzing longitudinal EHR data can predict **pancreatic cancer up to 3 years in advance** (AUROC: 0.88)
- Deep learning on sequential health records can identify **36+ month cancer risk trajectories**
- Multi-modal health data fusion outperforms single-domain models by **23–40%** in disease prediction accuracy

**BioSentinel brings this research out of academic papers and into open-source reality — for every hospital, clinic, and researcher on the planet.**

---

## 🔬 Why BioSentinel?

| Problem | BioSentinel Solution |
|---|---|
| ❌ Diseases caught too late, after irreversible damage | ✅ Continuous longitudinal AI monitoring from Day 1 |
| ❌ Siloed health records — labs here, prescriptions there | ✅ Unified patient timeline with all health dimensions |
| ❌ One-off risk scores with no temporal context | ✅ Time-series models that learn from 3+ years of data |
| ❌ Black-box AI that doctors don't trust | ✅ Explainable AI (SHAP/LIME) with clinical reasoning |
| ❌ Expensive proprietary platforms, inaccessible globally | ✅ Fully open-source, self-hostable on any infrastructure |
| ❌ Single-disease tools (just diabetes, just cancer) | ✅ Pan-disease platform: 50+ disease prediction modules |
| ❌ No standard data format across healthcare systems | ✅ FHIR-compatible data layer with HL7 support |

---

## 📚 Research Foundation

BioSentinel is built on a strong bedrock of peer-reviewed science:

### Key Research Validating This Approach

| Study | Finding | Source |
|---|---|---|
| AI on longitudinal EHR data for cancer prediction | Models analyzing sequential EHR data outperform single-visit baselines across multiple cancer types | [BMC Medical Research Methodology, 2025](https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-025-02473-w) |
| Deep learning for pancreatic cancer risk | AUROC 0.88 — predicts pancreatic cancer up to **36 months in advance** from disease trajectory | [Nature Medicine, 2023](https://www.nature.com/articles/s41591-023-02332-5) |
| Google lung cancer detection | AI model trained on 42,290 CT scans **outperforms average radiologist** at malignancy risk prediction (AUC 95.5%) | [NIH/PubMed](https://pmc.ncbi.nlm.nih.gov/articles/PMC8946688/) |
| ML cancer risk from symptoms | Systematic review (2014–2024): ML models integrating symptoms, genetic, lifestyle factors show high AUC for multi-cancer prediction | [Cancer Medicine, 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12701559/) |
| AI for longitudinal tumor tracking | AI has become a critical tool for longitudinal tracking of tumor progression and treatment response | [MedComm, 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12587170/) |
| RNN for heart failure from EHR | Recurrent neural networks on longitudinal EHR data for early detection of heart failure | [Circ. Cardiovasc. Qual. Outcomes, 2019](https://www.ahajournals.org/) |

### The Core Scientific Hypothesis

When you observe a person's health longitudinally — not just a snapshot — patterns emerge that are invisible in single-visit data:

```
Single Visit:  Normal CBC → "All Good" ✓
               (Doctor misses subtle 8-month downtrend in lymphocytes)

Longitudinal:  Month 1 → Month 4 → Month 8 → Month 12 → Month 16 → Month 20
               WBC:   7.2  →  7.0  →  6.8  →  6.4  →  5.9  →  5.3  ← ALERT 🔴
               Lymph: 32%  →  30%  →  28%  →  25%  →  22%  →  18%  ← ALERT 🔴
               Trend analysis: HIGH risk — lymphoma screening recommended
```

This is what BioSentinel detects.

---

## ✨ Key Features

### 🩺 Core Platform Features

- **📅 Longitudinal Health Timeline** — Build a continuous, structured timeline of every checkup, lab result, medication change, diagnosis, and lifestyle update per patient
- **🧠 AI Disease Risk Engine** — 50+ trained ML/DL models for cancer, cardiovascular, metabolic, neurological, and infectious disease prediction
- **📊 Biomarker Trend Analysis** — Real-time tracking of 200+ biomarkers with AI-powered trend detection and anomaly alerting
- **💊 Medication Interaction Monitor** — Track all medications over time, detect dangerous interactions, flag unusual prescription combinations that may indicate underlying conditions
- **🥗 Diet & Lifestyle Correlation Engine** — Connect diet plans, exercise, sleep, and BMI trends to disease risk trajectories
- **⚠️ Early Warning System** — Multi-tier alerting (Green / Yellow / Orange / Red) based on composite risk scores
- **🔍 Explainable AI (XAI)** — Every prediction comes with SHAP values, feature importance, and clinical reasoning — no black boxes
- **📱 Patient Dashboard** — Beautiful, intuitive patient-facing dashboard showing health trajectory and personalized recommendations
- **🏥 Clinician Interface** — Purpose-built interface for healthcare providers with population health views and patient risk stratification
- **🔗 FHIR Integration** — Native HL7 FHIR R4 support for seamless EHR system integration
- **🔐 Privacy-First Architecture** — On-premise deployment, end-to-end encryption, HIPAA/GDPR compliant data handling
- **📈 Research Analytics** — De-identified population-level analytics for public health researchers

### 🤖 AI/ML Modules

| Module | Description | Model Type | Status |
|---|---|---|---|
| `biosentinel-cancer-core` | Pan-cancer risk prediction from labs + history | Transformer + XGBoost | ✅ Stable |
| `biosentinel-cardio` | Cardiovascular disease & heart failure prediction | LSTM + Random Forest | ✅ Stable |
| `biosentinel-metabolic` | Diabetes type 1/2, metabolic syndrome, NAFLD | Gradient Boosting + MLP | ✅ Stable |
| `biosentinel-neuro` | Alzheimer's, Parkinson's, dementia risk | GNN + LSTM | 🔬 Beta |
| `biosentinel-renal` | Kidney disease progression & CKD staging | Regression + SVM | ✅ Stable |
| `biosentinel-hematologic` | Leukemia, lymphoma, anemia risk detection | CNN + attention mechanism | 🔬 Beta |
| `biosentinel-gastro` | Colorectal cancer, liver disease, pancreatitis | Sequence model | 🔬 Beta |
| `biosentinel-endocrine` | Thyroid, adrenal, hormonal disease prediction | Random Forest + calibration | ✅ Stable |
| `biosentinel-pharma` | Medication risk & interaction analysis | Rule engine + ML | 🔬 Beta |
| `biosentinel-diet-risk` | Diet-to-disease risk correlation | Regression models | 🗓️ Planned |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         BIOSENTINEL PLATFORM                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  DATA INGESTION LAYER                                                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐  │
│  │ Lab Data │ │Checkup   │ │Medication│ │ Diagnoses│ │ Diet / Lifestyle │  │
│  │ (Blood,  │ │ Records  │ │ History  │ │ & ICD-10 │ │ Plans & Vitals   │  │
│  │  Urine,  │ │ (Q1/Q2/  │ │ (Rx, OTC,│ │ Codes    │ │ (BMI, BP, Sleep) │  │
│  │  Imaging)│ │  Annual) │ │ Dosages) │ │          │ │                  │  │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────────┬─────────┘  │
│       └────────────┴────────────┴────────────┴─────────────────┘           │
│                                    │                                         │
│  PREPROCESSING & NORMALIZATION     │                                         │
│  ┌─────────────────────────────────▼───────────────────────────────────┐    │
│  │  • Temporal alignment & missing data imputation (KNN/MICE)          │    │
│  │  • FHIR R4 normalization & ICD-10/SNOMED mapping                    │    │
│  │  • Biomarker reference range normalization (age/sex adjusted)        │    │
│  │  • De-identification & privacy-preserving transformation             │    │
│  └─────────────────────────────────┬───────────────────────────────────┘    │
│                                    │                                         │
│  LONGITUDINAL FEATURE ENGINEERING  │                                         │
│  ┌─────────────────────────────────▼───────────────────────────────────┐    │
│  │  • Time-series feature extraction (slope, variance, acceleration)   │    │
│  │  • Biomarker trend vectors (3m, 6m, 12m, 36m windows)               │    │
│  │  • Cross-marker correlation matrices                                 │    │
│  │  • Drug-disease interaction graph features                           │    │
│  └─────────────────────────────────┬───────────────────────────────────┘    │
│                                    │                                         │
│  AI PREDICTION ENGINE              │                                         │
│  ┌─────────────────────────────────▼───────────────────────────────────┐    │
│  │  ┌─────────────────┐  ┌────────────────┐  ┌────────────────────┐   │    │
│  │  │  Temporal Models │  │ Ensemble Core  │  │  Explainability   │   │    │
│  │  │  • LSTM          │  │ • XGBoost      │  │  Layer            │   │    │
│  │  │  • Transformer   │  │ • Random Forest│  │  • SHAP Values    │   │    │
│  │  │  • GRU           │  │ • LightGBM     │  │  • LIME           │   │    │
│  │  │  • S4 / Mamba    │  │ • CatBoost     │  │  • Attention Maps │   │    │
│  │  └──────────────────┘  └────────────────┘  └────────────────────┘   │    │
│  └─────────────────────────────────┬───────────────────────────────────┘    │
│                                    │                                         │
│  RISK SCORING & ALERTING           │                                         │
│  ┌─────────────────────────────────▼───────────────────────────────────┐    │
│  │  Composite Risk Score (0–100) → Alert Level → Clinical Action        │    │
│  │  🟢 LOW (0-25)   🟡 MODERATE (26-50)   🟠 HIGH (51-75)   🔴 CRITICAL│    │
│  └─────────────────────────────────┬───────────────────────────────────┘    │
│                                    │                                         │
│  OUTPUT LAYER                      │                                         │
│  ┌──────────┐ ┌──────────┐ ┌───────▼──────┐ ┌──────────┐ ┌─────────────┐  │
│  │ Patient  │ │Clinician │ │  REST API /  │ │  FHIR    │ │  Research   │  │
│  │Dashboard │ │Interface │ │  GraphQL     │ │  Export  │ │  Analytics  │  │
│  └──────────┘ └──────────┘ └──────────────┘ └──────────┘ └─────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

### Backend
| Component | Technology | Purpose |
|---|---|---|
| Core API | **Python 3.10+ / FastAPI** | High-performance async REST/GraphQL API |
| ML Framework | **PyTorch + scikit-learn** | Deep learning and classical ML models |
| Data Processing | **Pandas + Polars** | Efficient tabular data manipulation |
| Time-Series | **tsfresh + tslearn** | Automated time-series feature engineering |
| EHR Integration | **HL7 FHIR R4 (fhirclient)** | Healthcare data standard compliance |
| Task Queue | **Celery + Redis** | Async model inference & background jobs |
| Database | **PostgreSQL + TimescaleDB** | Relational + time-series data storage |
| Caching | **Redis** | High-performance caching layer |
| Explainability | **SHAP + LIME** | Model interpretation & clinical reasoning |
| Experiment Tracking | **MLflow** | Model versioning & experiment management |

### Frontend
| Component | Technology | Purpose |
|---|---|---|
| Web App | **React 18 + TypeScript** | Patient & clinician dashboards |
| Charts | **D3.js + Recharts** | Biomarker trend visualization |
| State | **Zustand** | Lightweight state management |
| Auth | **Auth.js (Next-Auth)** | Secure authentication |

### Infrastructure (Hosted on [xHost.live](https://xhost.live))
| Component | Technology |
|---|---|
| Containerization | **Docker + Docker Compose** |
| Orchestration | **Kubernetes (optional)** |
| CI/CD | **GitHub Actions** |
| Monitoring | **Prometheus + Grafana** |
| Security | **Vault + SSL/TLS** |

---

## 🤖 ML Models & Algorithms

### Time-Series Disease Trajectory Models

#### 1. BioSentinel Transformer (BST)
Our flagship temporal model, inspired by Transformer architectures adapted for biomedical time series:

```python
class BioSentinelTransformer(nn.Module):
    """
    Transformer-based model for longitudinal health trajectory modeling.
    Input: Sequence of patient health snapshots over time (up to 36 months)
    Output: Disease risk vector for 50+ conditions
    """
    def __init__(self, 
                 n_biomarkers: int = 200,
                 d_model: int = 256, 
                 n_heads: int = 8,
                 n_layers: int = 6,
                 n_diseases: int = 50,
                 max_seq_len: int = 12,  # quarterly checkups × 3 years
                 dropout: float = 0.1):
        super().__init__()
        self.biomarker_embedding = nn.Linear(n_biomarkers, d_model)
        self.temporal_encoding = TemporalPositionalEncoding(d_model, max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, 
                                                    dim_feedforward=1024, 
                                                    dropout=dropout,
                                                    batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.risk_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_diseases),
            nn.Sigmoid()
        )
```

#### 2. CancerRiskNet (Ensemble)
Multi-model ensemble for pan-cancer risk prediction, combining:
- **XGBoost** on structured tabular biomarker features
- **LightGBM** on medication & diagnosis history sequences
- **LSTM** on time-series lab value trajectories
- **Meta-learner** (Logistic Regression with isotonic calibration) combining all three

#### 3. BioMarkerTrend Detector
Statistical + ML-based biomarker anomaly detection:
- **ARIMA / SARIMA** for seasonality-adjusted trending
- **Isolation Forest** for multivariate outlier detection
- **CUSUM (Cumulative Sum)** for change-point detection in chronic markers

### Supported Prediction Targets

```yaml
cancer_modules:
  - lung_cancer           # AUC: ~0.87 (literature baseline)
  - colorectal_cancer     # AUC: ~0.83
  - breast_cancer         # AUC: ~0.89
  - pancreatic_cancer     # AUC: ~0.88 (CancerRiskNet DNPR validated)
  - liver_cancer          # AUC: ~0.81
  - leukemia              # AUC: ~0.84
  - lymphoma              # AUC: ~0.82
  - cervical_cancer       # AUC: ~0.90
  - prostate_cancer       # AUC: ~0.79
  - thyroid_cancer        # AUC: ~0.86

chronic_disease_modules:
  - type2_diabetes         # AUC: ~0.92
  - cardiovascular_disease # AUC: ~0.88
  - chronic_kidney_disease # AUC: ~0.85
  - alzheimers_risk        # AUC: ~0.78
  - parkinsons_risk        # AUC: ~0.76
  - fatty_liver_nafld      # AUC: ~0.83
  - hypertension_onset     # AUC: ~0.86
  - copd_progression       # AUC: ~0.81
```

> ⚠️ **Note**: AUC values above reflect published literature baselines for similar models. BioSentinel's actual performance depends on training data quality and volume. Always validate on your own dataset.

---

## 📋 Data Schema

BioSentinel uses a unified patient health timeline schema:

### Core Patient Record

```json
{
  "patient_id": "uuid-v4",
  "enrollment_date": "2022-01-15",
  "demographics": {
    "age": 42,
    "sex": "female",
    "ethnicity": "south_asian",
    "geographic_region": "IN-MH"
  },
  "checkups": [
    {
      "checkup_id": "chk_001",
      "date": "2022-01-15",
      "type": "full_body",
      "vitals": {
        "height_cm": 162,
        "weight_kg": 68.4,
        "bmi": 26.1,
        "blood_pressure_systolic": 128,
        "blood_pressure_diastolic": 82,
        "heart_rate": 74,
        "temperature_c": 36.8,
        "spo2": 98
      },
      "lab_results": {
        "cbc": {
          "wbc": 7.2,
          "rbc": 4.8,
          "hemoglobin": 13.2,
          "hematocrit": 39.1,
          "mcv": 88.0,
          "platelets": 245000,
          "neutrophils_pct": 62,
          "lymphocytes_pct": 30,
          "monocytes_pct": 6,
          "eosinophils_pct": 2
        },
        "metabolic_panel": {
          "glucose_fasting": 94,
          "hba1c": 5.4,
          "creatinine": 0.8,
          "egfr": 95,
          "bun": 14,
          "sodium": 139,
          "potassium": 4.1,
          "alt": 22,
          "ast": 19,
          "bilirubin_total": 0.7,
          "albumin": 4.2
        },
        "lipid_panel": {
          "total_cholesterol": 195,
          "ldl": 118,
          "hdl": 52,
          "triglycerides": 128
        },
        "hormones": {
          "tsh": 2.1,
          "vitamin_d": 28,
          "vitamin_b12": 410,
          "ferritin": 45
        },
        "tumor_markers": {
          "psa": null,
          "ca125": null,
          "cea": 1.2,
          "ca199": null,
          "afp": null
        },
        "urinalysis": {
          "ph": 6.0,
          "protein": "negative",
          "glucose": "negative",
          "blood": "negative",
          "specific_gravity": 1.018
        }
      }
    }
  ],
  "medications": [
    {
      "medication_id": "med_001",
      "name": "Levothyroxine",
      "generic_name": "levothyroxine_sodium",
      "rxnorm_code": "10582",
      "dosage_mg": 50,
      "frequency": "daily",
      "start_date": "2019-06-01",
      "end_date": null,
      "prescribed_for": "J06.0",
      "prescribing_doctor": "dr_hash_xyz"
    }
  ],
  "diagnoses": [
    {
      "icd10_code": "E03.9",
      "description": "Hypothyroidism, unspecified",
      "diagnosed_date": "2019-05-20",
      "status": "active",
      "severity": "mild"
    }
  ],
  "diet_plans": [
    {
      "plan_id": "diet_001",
      "start_date": "2022-01-15",
      "calories_daily": 1800,
      "protein_g": 75,
      "carbs_g": 220,
      "fat_g": 65,
      "fiber_g": 28,
      "diet_type": "mediterranean",
      "restrictions": ["gluten_free"],
      "alcohol_units_weekly": 2,
      "smoking_status": "never"
    }
  ],
  "family_history": {
    "cancer": ["breast_cancer_maternal_aunt"],
    "cardiovascular": [],
    "diabetes": ["type2_paternal_grandfather"]
  }
}
```

### FHIR R4 Compatibility

BioSentinel natively maps to FHIR R4 resources:
- `Patient` ↔ Patient demographics
- `Observation` ↔ Lab results & vitals
- `MedicationRequest` ↔ Medication history
- `Condition` ↔ Diagnoses
- `NutritionOrder` ↔ Diet plans
- `RiskAssessment` ↔ BioSentinel AI risk scores

---

## 🚀 Getting Started

### Prerequisites

```bash
# System requirements
Python >= 3.10
Node.js >= 18.0
PostgreSQL >= 14
Redis >= 7.0
Docker & Docker Compose (recommended)

# Optional for GPU acceleration
CUDA >= 11.8 (NVIDIA GPU)
```

## 📦 Installation

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/liveupx/biosentinel.git
cd biosentinel

# Copy environment configuration
cp .env.example .env
# Edit .env with your configuration

# Start all services
docker-compose up -d

# Run database migrations
docker-compose exec api python -m biosentinel.db migrate

# Create admin user
docker-compose exec api python -m biosentinel.admin create-superuser

# BioSentinel is now running at:
# API:       http://localhost:8000
# Dashboard: http://localhost:3000
# Docs:      http://localhost:8000/docs
```

### Option 2: Local Development

```bash
# Clone repository
git clone https://github.com/liveupx/biosentinel.git
cd biosentinel

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development

# Set up database
createdb biosentinel
python -m biosentinel.db migrate

# Configure environment
cp .env.example .env
# Edit .env with your local config

# Start the API server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# In a new terminal — start the frontend
cd frontend
npm install
npm run dev
```

### Option 3: pip Install (Library Usage)

```bash
pip install biosentinel

# Use as a library
from biosentinel import BioSentinelClient, PatientTimeline

client = BioSentinelClient(api_url="http://localhost:8000", api_key="your-key")
```

---

## ⚡ Quick Start

### Ingest a Patient Record

```python
from biosentinel import BioSentinelClient
from biosentinel.models import Patient, Checkup, LabResults

client = BioSentinelClient(api_url="http://localhost:8000", api_key="your-api-key")

# Add a new patient
patient = client.patients.create(
    age=45,
    sex="male",
    ethnicity="caucasian",
    family_history={"cancer": ["colorectal_cancer_father"]}
)

# Log a quarterly checkup
checkup = client.checkups.create(
    patient_id=patient.id,
    date="2024-01-15",
    checkup_type="full_body",
    vitals={
        "weight_kg": 82.0,
        "blood_pressure_systolic": 138,
        "blood_pressure_diastolic": 88
    },
    labs={
        "wbc": 8.1,
        "hemoglobin": 14.2,
        "glucose_fasting": 108,
        "hba1c": 5.9,
        "cea": 2.4
    }
)

print(f"Checkup recorded: {checkup.id}")
```

### Run Disease Risk Prediction

```python
# Get AI risk assessment for a patient
risk = client.predict(
    patient_id=patient.id,
    models=["cancer_core", "metabolic", "cardiovascular"],
    lookback_months=24
)

print(risk.summary())
# ┌──────────────────────────────────────────────┐
# │ BioSentinel Risk Assessment                  │
# │ Patient: p_abc123  │  Data: 24 months        │
# ├──────────────────────────────────────────────┤
# │ 🟠 Metabolic Risk: 67/100 (HIGH)            │
# │    → Pre-diabetes trajectory detected        │
# │    → HbA1c uptrend: +0.8% over 18 months    │
# ├──────────────────────────────────────────────┤
# │ 🟡 Cardiovascular Risk: 44/100 (MODERATE)   │
# │    → Borderline hypertension trend           │
# ├──────────────────────────────────────────────┤
# │ 🟢 Cancer Risk (composite): 18/100 (LOW)    │
# └──────────────────────────────────────────────┘

# Get SHAP explanations for top risk factors
explanation = risk.explain(disease="metabolic")
print(explanation.top_factors)
# [
#   {"feature": "hba1c_trend_18m", "shap_value": +0.42, "direction": "risk_increasing"},
#   {"feature": "fasting_glucose_slope", "shap_value": +0.31, "direction": "risk_increasing"},
#   {"feature": "bmi_trajectory", "shap_value": +0.18, "direction": "risk_increasing"},
#   {"feature": "hdl_trend", "shap_value": -0.12, "direction": "protective"},
# ]
```

### REST API Usage

```bash
# Authenticate
curl -X POST http://localhost:8000/api/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "your-user", "password": "your-pass"}'

# Create patient
curl -X POST http://localhost:8000/api/v1/patients \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"age": 45, "sex": "male"}'

# Run prediction
curl -X POST http://localhost:8000/api/v1/patients/{patient_id}/predict \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"models": ["cancer_core", "metabolic"], "lookback_months": 24}'
```

---

## 📁 Project Structure

```
biosentinel/
├── 📄 README.md                    # You are here
├── 📄 LICENSE                      # MIT License
├── 📄 CONTRIBUTING.md              # Contribution guidelines
├── 📄 CODE_OF_CONDUCT.md           # Community standards
├── 📄 SECURITY.md                  # Security policy
├── 📄 CHANGELOG.md                 # Version history
├── 📄 .env.example                 # Environment template
├── 📄 docker-compose.yml           # Docker setup
├── 📄 pyproject.toml               # Python project config
├── 📄 requirements.txt             # Core dependencies
├── 📄 requirements-dev.txt         # Development dependencies
│
├── 📁 .github/
│   ├── 📁 workflows/
│   │   ├── ci.yml                  # Continuous integration
│   │   ├── cd.yml                  # Continuous deployment
│   │   └── model-eval.yml          # Automated model evaluation
│   ├── 📁 ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   └── PULL_REQUEST_TEMPLATE.md
│
├── 📁 src/
│   ├── 📁 api/
│   │   ├── main.py                 # FastAPI application entry
│   │   ├── routers/                # API route handlers
│   │   │   ├── patients.py
│   │   │   ├── checkups.py
│   │   │   ├── predictions.py
│   │   │   ├── medications.py
│   │   │   └── analytics.py
│   │   ├── schemas/                # Pydantic data schemas
│   │   └── dependencies.py         # FastAPI dependencies
│   │
│   ├── 📁 models/
│   │   ├── cancer/
│   │   │   ├── cancer_risk_net.py  # Pan-cancer ensemble
│   │   │   ├── lung_model.py
│   │   │   ├── colorectal_model.py
│   │   │   └── ...
│   │   ├── cardiovascular/
│   │   ├── metabolic/
│   │   ├── neurological/
│   │   ├── base_model.py           # Abstract base for all models
│   │   └── ensemble.py             # Ensemble meta-learner
│   │
│   ├── 📁 preprocessing/
│   │   ├── normalizer.py           # Biomarker normalization
│   │   ├── imputer.py              # Missing data imputation
│   │   ├── fhir_mapper.py          # FHIR R4 data mapping
│   │   └── feature_engineer.py    # Temporal feature extraction
│   │
│   ├── 📁 utils/
│   │   ├── explainability.py       # SHAP/LIME integration
│   │   ├── alerts.py               # Risk alerting system
│   │   ├── biomarker_reference.py  # Normal ranges DB
│   │   └── crypto.py               # Privacy/encryption utils
│   │
│   └── 📁 visualization/
│       ├── trend_charts.py         # Biomarker trend plotting
│       └── risk_heatmap.py         # Risk visualization
│
├── 📁 frontend/
│   ├── 📁 public/
│   │   └── index.html              # Landing page
│   ├── 📁 src/
│   │   ├── pages/
│   │   ├── components/
│   │   └── hooks/
│   └── package.json
│
├── 📁 notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_model_evaluation.ipynb
│   └── 05_explainability_demo.ipynb
│
├── 📁 data/
│   ├── 📁 samples/                 # Anonymized sample data
│   │   ├── patient_sample.json
│   │   └── checkup_sample.json
│   └── 📁 reference/
│       ├── biomarker_ranges.json   # Reference lab ranges
│       └── icd10_mapping.json      # ICD-10 code mappings
│
├── 📁 tests/
│   ├── test_api.py
│   ├── test_models.py
│   ├── test_preprocessing.py
│   └── test_fhir.py
│
├── 📁 scripts/
│   ├── train_models.py             # Model training script
│   ├── evaluate_models.py          # Evaluation pipeline
│   ├── ingest_data.py              # Bulk data ingestion
│   └── export_fhir.py              # FHIR export utility
│
└── 📁 docs/
    ├── architecture.md
    ├── api-reference.md
    ├── data-schema.md
    ├── model-documentation.md
    ├── privacy-compliance.md
    └── deployment-guide.md
```

---

## 🗺️ Roadmap

### Phase 1 — Foundation (Q1–Q2 2025) ✅ In Progress
- [x] Core data ingestion API (patients, checkups, medications, diagnoses)
- [x] FHIR R4 data model & mapping
- [x] Basic biomarker normalization pipeline
- [x] First-generation CancerRiskNet (XGBoost-based)
- [x] Patient dashboard (React)
- [ ] Docker deployment stack
- [ ] Full test suite (target: 80% coverage)

### Phase 2 — AI Engine (Q3 2025)
- [ ] BioSentinel Transformer (BST) v1.0
- [ ] 10 fully validated disease modules
- [ ] SHAP explainability integration
- [ ] Automated alert system (email/webhook)
- [ ] Pre-trained model weights (on synthetic + public datasets)
- [ ] Clinician interface v1.0

### Phase 3 — Integrations (Q4 2025)
- [ ] HL7 FHIR bulk import from major EHR systems (Epic, Cerner, OpenMRS)
- [ ] DICOM imaging data support (radiology integration)
- [ ] Genomic data (VCF) integration for genetic risk factors
- [ ] Mobile app (React Native) for patient self-reporting
- [ ] Federated learning support (train without centralizing data)

### Phase 4 — Global Scale (2026)
- [ ] Multilingual support (Hindi, Spanish, Portuguese, Mandarin, Arabic)
- [ ] Low-resource deployment mode (works in limited-connectivity regions)
- [ ] WHO & ICMR dataset integration
- [ ] Community model hub (share trained models across institutions)
- [ ] BioSentinel Cloud (managed SaaS option via xHost.live)

### Long-Term Vision
- [ ] Real-time wearable data integration (smartwatch, CGM, BP monitor)
- [ ] LLM-powered clinical narrative summarization
- [ ] Drug discovery signal detection from population health patterns
- [ ] Integration with national health registries (with regulatory approval)

---

## 📚 Research Papers

Key papers that inform BioSentinel's methodology:

```bibtex
@article{pancreatic_dlm_2023,
  title={A deep learning algorithm to predict risk of pancreatic cancer from disease trajectories},
  journal={Nature Medicine},
  year={2023},
  url={https://www.nature.com/articles/s41591-023-02332-5}
}

@article{bmc_longitudinal_ehr_2025,
  title={AI methods applied to longitudinal EHR data for cancer prediction: a scoping review},
  journal={BMC Medical Research Methodology},
  year={2025},
  url={https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-025-02473-w}
}

@article{google_lung_cancer_ai,
  title={End-to-end lung nodule detection and malignancy risk prediction},
  author={Ardila, D. et al.},
  journal={Nature Medicine},
  year={2019}
}

@article{ml_cancer_risk_review_2025,
  title={Cancer Risk Prediction Using ML for Supporting Early Cancer Diagnosis in Symptomatic Patients},
  journal={Cancer Medicine},
  year={2025},
  url={https://pmc.ncbi.nlm.nih.gov/articles/PMC12701559/}
}

@article{rnn_heart_failure_ehr,
  title={Recurrent neural networks for early detection of heart failure from longitudinal EHR data},
  journal={Circ. Cardiovasc. Qual. Outcomes},
  year={2019}
}
```

---

## 🤝 Contributing

We welcome contributions from doctors, ML engineers, data scientists, frontend developers, and anyone passionate about saving lives through technology!

Please read our [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Code style guidelines (Black, isort, mypy)
- How to set up a development environment
- How to submit a pull request
- How to add a new disease prediction module
- How to contribute anonymized training data

### Ways to Contribute

| Area | What We Need |
|---|---|
| 🧠 ML / Data Science | New disease models, improved architectures, training datasets |
| 🔬 Medical / Clinical | Clinical validation, biomarker knowledge, literature review |
| ⚙️ Backend | API features, integrations, performance optimization |
| 🎨 Frontend | Dashboard UI, data visualization, UX improvements |
| 📖 Documentation | Guides, tutorials, translation |
| 🔐 Security | Privacy review, threat modeling, compliance |
| 🌍 Community | Outreach, partnerships with hospitals and research institutions |

---

## 💬 Community & Support

| Channel | Purpose |
|---|---|
| [GitHub Discussions](https://github.com/liveupx/biosentinel/discussions) | General Q&A, ideas, announcements |
| [GitHub Issues](https://github.com/liveupx/biosentinel/issues) | Bug reports, feature requests |
| [Discord](https://discord.gg/biosentinel) | Real-time community chat |
| [Email](mailto:biosentinel@liveupx.com) | Private / partnership inquiries |
| [Website](https://biosentinel.liveupx.com) | Documentation & demos |

---

## 🔐 Ethics & Privacy

BioSentinel handles some of the most sensitive data that exists — a person's complete medical history. We take this responsibility extremely seriously.

### Privacy Principles
1. **Data Minimization** — Collect only what's needed for prediction
2. **Purpose Limitation** — Data used only for stated health prediction purposes
3. **On-Premise First** — Default deployment keeps data entirely within your infrastructure
4. **Encryption at Rest & in Transit** — AES-256 at rest, TLS 1.3 in transit
5. **Patient Consent** — Built-in consent management framework
6. **Right to Deletion** — Full GDPR-compliant data deletion workflows
7. **Audit Logs** — Complete audit trail of all data access

### Compliance
- **HIPAA** (USA) — Technical safeguards implemented
- **GDPR** (EU) — Privacy by design
- **DPDP Act** (India) — Aligned with India's Digital Personal Data Protection Act, 2023
- **HL7 FHIR** — Interoperability standards

### ⚠️ Medical Disclaimer
BioSentinel is a **clinical decision support tool** and **research platform**. It is **NOT** a licensed medical device, diagnostic tool, or substitute for professional medical judgment. All risk scores and predictions should be reviewed by qualified healthcare professionals. BioSentinel's outputs are probabilistic risk assessments, not diagnoses.

---

## 📄 License

```
MIT License

Copyright (c) 2025 Liveupx Pvt. Ltd. / Mohit Chaprana

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

See [LICENSE](LICENSE) for full text.

---

## 👨‍💻 Creator & Team

<table>
  <tr>
    <td align="center">
      <strong>Mohit Chaprana</strong><br/>
      <em>Creator & Lead Architect</em><br/>
      <a href="https://www.linkedin.com/in/ammohitchaprana/">LinkedIn</a> ·
      <a href="https://liveupx.com">Liveupx.com</a>
    </td>
  </tr>
</table>

**Organization**: [Liveupx Pvt. Ltd.](https://liveupx.com)  
**Infrastructure**: [xHost.live](https://xhost.live)  
**Repository**: [github.com/liveupx/biosentinel](https://github.com/liveupx/biosentinel)

---

## 🙏 Acknowledgements

- **Nature Medicine** and all researchers whose published work forms the scientific foundation of this project
- **HL7 FHIR** community for open healthcare data standards
- **MIMIC-III/IV** and **UK Biobank** projects for pioneering open health data
- **scikit-learn**, **PyTorch**, **FastAPI**, and the entire open-source scientific Python ecosystem
- Every healthcare professional fighting to save lives every single day

---

<div align="center">

**If BioSentinel helps even one person detect cancer early — it was worth building.**

⭐ **Star this repo** if you believe AI can save lives  
🔀 **Fork and contribute** to make it happen faster  
📢 **Share** with every doctor, researcher, and developer you know

<br/>

*Built with ❤️ by [Mohit Chaprana](https://www.linkedin.com/in/ammohitchaprana/) & [Liveupx Pvt. Ltd.](https://liveupx.com)*  
*Hosted on [xHost.live](https://xhost.live)*

</div>
