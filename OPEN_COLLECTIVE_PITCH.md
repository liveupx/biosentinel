# BioSentinel — Open Source AI Health Monitoring

> *"The best time to catch a serious disease is before you feel sick."*

## What we built

BioSentinel is an open-source AI platform that tracks a person's complete health
journey across quarterly checkups — detecting the subtle rising trends in blood
sugar, tumour markers, and immune cells that signal cancer, diabetes, and heart
disease **years before symptoms appear.**

Most health apps show you whether today's test result is normal or abnormal.
BioSentinel does something different: it watches the *direction* of your results
over 12–36 months.

| What a clinician sees | What BioSentinel detects |
|---|---|
| HbA1c = 5.9% — "borderline, watch it" | 5.5 → 5.6 → 5.7 → 5.8 → 5.9 over 24 months = **pre-diabetes trajectory** |
| CEA = 3.2 ng/mL — "within limits" | 1.5 → 1.9 → 2.3 → 2.8 → 3.2 over 18 months = **rising tumour marker** |
| Lymphocytes = 24% — "low-normal" | 32% → 29% → 27% → 25% → 24% = **immune decline, flag for review** |

---

## Why this matters

A 2023 paper in *Nature Medicine* showed that AI trained on 6 million patient
records can predict pancreatic cancer **36 months before diagnosis** with an
AUROC of 0.88. A Google/NIH study achieved AUC 95.5% on lung cancer CT scans —
outperforming average radiologists.

The core insight from all this research: **a single test result tells you almost
nothing. A trend tells you everything.**

BioSentinel brings this approach to general practice — not just oncology centres.

---

## What's built (v2.1.0)

- **4 machine learning models** (GradientBoosting + real SHAP explanations) across
  cancer, metabolic, cardiovascular, and hematologic risk domains
- **37 REST API endpoints** — full patient management, checkup ingestion, predictions,
  alerts, medications, diagnoses, diet plans, audit log
- **Claude AI integration** — upload a photo of any lab report and it auto-fills
  the checkup form (Claude Vision); generates plain-English summaries for patients;
  detects longitudinal anomalies that fixed thresholds miss
- **Patient self-service portal** — patients can view their own trends in plain
  language, without needing to understand medical terminology
- **Background scheduler** — automatically emails clinicians when a patient is
  overdue for their quarterly checkup
- **250+ passing tests** with a clean CI pipeline on Python 3.10/3.11/3.12
- **MIT licensed** — free to use, modify, and deploy

---

## Who this is for

- **Clinicians in lower-resource settings** — a single doctor in a district hospital
  who manages hundreds of patients and cannot track trends manually
- **Preventive health clinics** — annual health check services that want to
  provide more than a "normal/abnormal" result
- **Medical researchers** — a platform for studying longitudinal biomarker
  trajectories at population scale
- **Healthcare students** — learning ML-in-medicine with a real, working codebase

---

## What your sponsorship funds

BioSentinel is maintained by a solo developer ([Mohit Chaprana](https://liveupx.com))
as an open-source project alongside commercial work. Sponsorship goes toward:

| Priority | Item | Estimated effort |
|---|---|---|
| 🔴 **Immediate** | MIMIC-IV dataset access + retraining on 65K real hospital records | 4–6 weeks |
| 🔴 **Immediate** | Clinical validation study with a partner hospital | 2–3 months |
| 🟡 **This year** | Real SHAP on MIMIC-IV data — publish AUC metrics openly | 2 weeks |
| 🟡 **This year** | Mobile-responsive React Native / PWA wrapper | 4–6 weeks |
| 🔵 **Next year** | Federated learning — clinics train a shared model without sharing data | 3 months |

**Sponsorship goal: $500/month**

This covers:
- Server costs for a hosted demo instance (currently self-hosted)
- Time to maintain the API and test suite as FastAPI/scikit-learn evolve
- Time to conduct the MIMIC-IV retraining and publish results

---

## Current limitations (we're honest about these)

The ML models are currently trained on **synthetic data** — they are directionally
correct but not clinically validated. A score of 60% cancer risk does not mean
the patient has a 60% chance of getting cancer. It means their biomarker pattern
resembles patterns associated with elevated risk in the training data.

**Clinical validation on real patient data is the #1 priority**, and it is what
sponsorship will directly fund.

---

## Get involved

**Use it:** github.com/liveupx/biosentinel  
**Star it:** Every star helps with visibility in healthcare IT communities  
**Contribute:** Especially welcome — clinicians, ML engineers, medical informaticists  
**Sponsor:** opencollective.com/biosentinel

If you work in healthcare and want to pilot BioSentinel at your clinic, please
reach out: mohit@liveupx.com

---

*BioSentinel is a research and clinical decision-support platform. It is NOT a
licensed medical device and does NOT replace qualified medical professionals.
All AI outputs must be reviewed by licensed healthcare providers before any
clinical action is taken.*

*MIT License · Liveupx Pvt. Ltd. · github.com/liveupx/biosentinel*
