# BioSentinel Architecture

## Overview

BioSentinel is a multi-layer platform built around a core principle: **time is the most valuable dimension in health data**. Traditional clinical tools analyze a snapshot of a patient's health. BioSentinel analyzes the full trajectory.

---

## Layers

### 1. Data Ingestion Layer

Accepts health data through:
- **REST API** — structured JSON input (primary interface)
- **FHIR R4 Bulk Import** — from hospital EHR systems (Epic, Cerner, OpenMRS)
- **CSV Import** — for research data ingestion
- **Direct SDK** — Python client library for integration

Data types accepted:
- Lab results (CBC, metabolic panel, lipids, hormones, tumor markers, urinalysis)
- Vitals (BP, weight, BMI, heart rate, SpO2, temperature)
- Medications (name, dosage, frequency, start/end dates, indications)
- Diagnoses (ICD-10 codes, severity, status)
- Diet plans (macronutrients, diet type, restrictions, alcohol, smoking)
- Family history (genetic risk factors)

### 2. Preprocessing & Normalization

Before features reach the ML models, data undergoes:

**Temporal Alignment**: All checkups are aligned to a consistent time axis. Irregular checkup timing is handled by interpolation or forward-fill where appropriate.

**Reference Range Normalization**: Biomarker values are normalized by age/sex-adjusted reference ranges. A hemoglobin of 12 g/dL means something different for a 25-year-old woman vs. a 65-year-old man.

**Missing Data Imputation**: KNN-based imputation for missing lab values when sufficient neighboring checkups exist. MICE (Multiple Imputation by Chained Equations) for more complex missingness patterns.

**FHIR Mapping**: All internal data structures can be bidirectionally mapped to FHIR R4 resources.

**De-identification**: Patient IDs are pseudonymized UUIDs. PHI stripping is available for research/analytics use cases.

### 3. Temporal Feature Engineering

This is BioSentinel's core differentiator. Raw biomarker values are transformed into rich temporal features:

**Trend Features** (per biomarker, per time window: 3m, 6m, 12m, 24m, 36m):
- Linear trend slope (rate of change)
- Acceleration (second derivative)
- Variance / standard deviation
- Min/max within window
- Crossing of reference range boundaries

**Cross-Marker Features**:
- Correlation matrices between related biomarker groups
- Ratio trends (e.g., LDL/HDL ratio trajectory)
- Composite index trends (e.g., HOMA-IR for insulin resistance)

**Clinical Event Features**:
- New medication introductions
- Medication dose changes
- New diagnoses
- Diet plan changes

The result is ~1,500 temporal features per patient per prediction run.

### 4. AI Prediction Engine

Three model architectures work in concert:

**BioSentinel Transformer (BST)**
- Adapted Transformer encoder for biomedical time series
- Input: sequence of health snapshots (up to 12 quarterly checkups = 3 years)
- Self-attention across time steps and biomarker dimensions
- Pre-trained on synthetic longitudinal data; fine-tunable on real cohorts

**CancerRiskNet Ensemble**
- XGBoost on tabular temporal features (primary model)
- LightGBM on medication/diagnosis sequence features
- LSTM on raw biomarker time series
- Logistic regression meta-learner combining all three outputs

**BioMarkerTrend Detector**
- CUSUM change-point detection for chronic biomarker monitoring
- Isolation Forest for multivariate anomaly detection
- ARIMA for seasonality-adjusted trending

### 5. Risk Scoring & Explainability

Every prediction produces:
- **Risk Score**: Calibrated probability 0.0–1.0 (mapped to 0–100 display scale)
- **Risk Level**: LOW / MODERATE / HIGH / CRITICAL
- **SHAP Values**: Per-feature contribution to the prediction
- **Top Factors**: Human-readable explanation of top 5 risk-increasing features
- **Data Completeness Score**: How much of the expected feature set was available
- **Clinical Recommendation**: Suggested next steps

### 6. Alert & Output Layer

- **REST API**: All predictions available via authenticated REST endpoints
- **WebSocket**: Real-time alert streaming for clinical dashboards
- **Email Alerts**: Configurable email notifications for HIGH/CRITICAL predictions
- **Webhooks**: POST risk alerts to external systems
- **FHIR RiskAssessment**: Export predictions as FHIR R4 `RiskAssessment` resources
- **PDF Reports**: Automated patient risk report generation

---

## Data Flow Diagram

```
Patient Checkup Data
        │
        ▼
┌──────────────────┐
│  Ingestion API   │  ← REST, FHIR, CSV, SDK
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Preprocessing   │  ← Normalize, Impute, Align
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Feature Eng.    │  ← ~1,500 temporal features
└────────┬─────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐ ┌──────────┐
│  BST  │ │CancerRisk│  ← Disease-specific models
└───┬───┘ │  Net     │
    │     └─────┬────┘
    └─────┬─────┘
          │
          ▼
┌──────────────────┐
│  Risk Scoring    │  ← Calibration + SHAP
│  + Explainability│
└────────┬─────────┘
         │
    ┌────┴──────┬──────────┐
    ▼           ▼          ▼
 Dashboard   Alerts    FHIR Export
```

---

## Security Architecture

See [privacy-compliance.md](privacy-compliance.md) for full details.

- TLS 1.3 enforced for all API communication
- AES-256 encryption at rest (configurable)
- JWT authentication with short-lived access tokens
- Role-based access control (Patient / Clinician / Admin / Researcher)
- Row-level security in PostgreSQL for multi-tenant deployments
- Complete audit log of all patient data access
- Patient IDs are pseudonymized UUIDs (never sequential integers)
