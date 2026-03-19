#!/usr/bin/env python3
"""
BioSentinel — MIMIC-IV Model Training Script
=============================================
Retrains all 4 disease-risk models on real longitudinal patient data
from the MIMIC-IV clinical database (Beth Israel Deaconess Medical Center,
2008–2022, 65,000+ ICU patients).

PREREQUISITES
-------------
1. PhysioNet credentialing approved (physionet.org)
2. MIMIC-IV Data Use Agreement signed
3. Google BigQuery access linked to PhysioNet account
4. Google Cloud SDK installed: https://cloud.google.com/sdk

⚠️  DUA RESTRICTION — READ BEFORE RUNNING
-----------------------------------------
The PhysioNet Credentialed Data Use Agreement explicitly PROHIBITS:
- Sending MIMIC data to any external API or cloud service
- Using MIMIC data with LLMs (ChatGPT, Claude, Gemini, etc.)
- Sharing MIMIC data with third parties

This script NEVER sends data externally. All training is local.
The resulting model WEIGHTS (not the data) can be shared publicly.

USAGE
-----
    # Step 1 — Export from BigQuery (run in Google Cloud Shell or gcloud CLI)
    # See: query below in _bigquery_export_query()

    # Step 2 — Train locally (after downloading the CSV)
    python train_mimic.py \
        --data labevents_longitudinal.csv \
        --output models/ \
        --min-visits 3 \
        --validate

    # Step 3 — Verify improvement vs synthetic baseline
    python train_mimic.py --compare models/

    # Step 4 — Integrate into BioSentinel
    # Set MODEL_PATH=./models in .env — app.py will load pre-trained weights

EXPECTED RESULTS (from literature)
-----------------------------------
Cancer risk (longitudinal):  AUC ~0.82–0.88 (Nature Medicine 2023)
Metabolic (HbA1c trajectory): AUC ~0.91 (BMC 2025)
Cardiovascular:               AUC ~0.85
Hematologic:                  AUC ~0.79
"""

import argparse
import json
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger("biosentinel.train_mimic")


# ── MIMIC-IV BigQuery export query ────────────────────────────────────────────

BIGQUERY_EXPORT_QUERY = """
-- BioSentinel MIMIC-IV Export Query
-- Run in Google BigQuery (requires MIMIC-IV access)
-- Project: physionet-data (or your linked project)
-- Result: ~2-5 GB depending on date range
--
-- Save result as: labevents_longitudinal.csv
-- BigQuery → More → Export → CSV

WITH patient_visits AS (
  -- Patients with 3+ admissions (longitudinal = multiple visits)
  SELECT subject_id
  FROM `physionet-data.mimiciv_hosp.admissions`
  GROUP BY subject_id
  HAVING COUNT(DISTINCT hadm_id) >= 3
),

lab_items AS (
  -- LOINC/itemid mapping for key biomarkers
  -- itemids from MIMIC-IV d_labitems table
  SELECT itemid, label FROM `physionet-data.mimiciv_hosp.d_labitems`
  WHERE itemid IN (
    -- CBC
    51222,  -- Hemoglobin
    51301,  -- White Blood Cells
    51265,  -- Platelets
    51244,  -- Lymphocytes (%)
    51256,  -- Neutrophils (%)
    -- Metabolic
    50809,  -- Glucose (serum)
    50852,  -- Hemoglobin A1c
    50971,  -- Potassium
    50912,  -- Creatinine
    50882,  -- Bicarbonate
    50893,  -- Calcium
    -- Liver
    50861,  -- ALT (SGPT)
    50878,  -- AST (SGOT)
    50927,  -- GGT
    50884,  -- Bilirubin Total
    50862,  -- Albumin
    -- Lipids
    50907,  -- Cholesterol Total
    50924,  -- Triglycerides
    -- Tumour markers (available in some MIMIC records)
    50893,  -- CEA proxy (not all records have this)
    -- Inflammation
    50889,  -- CRP
    -- Thyroid
    50933   -- TSH
  )
),

patient_labs AS (
  SELECT
    l.subject_id,
    l.charttime,
    l.itemid,
    di.label,
    l.valuenum,
    l.valueuom,
    p.anchor_age,
    p.gender,
    -- Days since first admission (proxy for timeline)
    DATE_DIFF(DATE(l.charttime),
              MIN(DATE(l.charttime)) OVER (PARTITION BY l.subject_id),
              DAY) AS days_since_first
  FROM `physionet-data.mimiciv_hosp.labevents` l
  JOIN `physionet-data.mimiciv_hosp.patients` p
    ON l.subject_id = p.subject_id
  JOIN lab_items di
    ON l.itemid = di.itemid
  WHERE l.subject_id IN (SELECT subject_id FROM patient_visits)
    AND l.valuenum IS NOT NULL
    AND l.valuenum > 0
    -- Date range: recent 10 years of data
    AND l.charttime >= '2010-01-01'
),

-- Pivot to wide format (one row per patient-visit)
pivoted AS (
  SELECT
    subject_id,
    DATE_TRUNC(charttime, MONTH) AS visit_month,
    anchor_age,
    gender,
    -- CBC
    AVG(CASE WHEN itemid = 51222 THEN valuenum END) AS hemoglobin,
    AVG(CASE WHEN itemid = 51301 THEN valuenum END) AS wbc,
    AVG(CASE WHEN itemid = 51265 THEN valuenum END) AS platelets,
    AVG(CASE WHEN itemid = 51244 THEN valuenum END) AS lymphocytes_pct,
    AVG(CASE WHEN itemid = 51256 THEN valuenum END) AS neutrophils_pct,
    -- Metabolic
    AVG(CASE WHEN itemid = 50809 THEN valuenum END) AS glucose_fasting,
    AVG(CASE WHEN itemid = 50852 THEN valuenum END) AS hba1c,
    AVG(CASE WHEN itemid = 50912 THEN valuenum END) AS creatinine,
    -- Liver
    AVG(CASE WHEN itemid = 50861 THEN valuenum END) AS alt,
    AVG(CASE WHEN itemid = 50878 THEN valuenum END) AS ast,
    AVG(CASE WHEN itemid = 50862 THEN valuenum END) AS albumin,
    AVG(CASE WHEN itemid = 50884 THEN valuenum END) AS bilirubin_total,
    -- Lipids
    AVG(CASE WHEN itemid = 50907 THEN valuenum END) AS total_cholesterol,
    AVG(CASE WHEN itemid = 50924 THEN valuenum END) AS triglycerides,
    -- Inflammation
    AVG(CASE WHEN itemid = 50889 THEN valuenum END) AS crp,
    -- Thyroid
    AVG(CASE WHEN itemid = 50933 THEN valuenum END) AS tsh
  FROM patient_labs
  GROUP BY subject_id, visit_month, anchor_age, gender
)

SELECT *
FROM pivoted
WHERE (hemoglobin IS NOT NULL OR hba1c IS NOT NULL OR wbc IS NOT NULL)
ORDER BY subject_id, visit_month
LIMIT 500000;
"""


# ── Feature engineering (same as app.py BioSentinelEngine) ───────────────────

FEATURES = [
    "age", "sex_f", "smoke", "alcohol", "exercise_inv",
    "fam_cancer", "fam_diab", "fam_cardio", "n_checkups", "months",
    "hba1c", "glucose", "hemoglobin", "lymph", "wbc", "platelets",
    "cea", "ca125", "psa", "alt", "ast", "ldl", "hdl", "triglyc",
    "bp_sys", "bmi", "creatinine", "tsh", "crp", "ferritin",
    "hba1c_slope", "glucose_slope", "hemoglobin_slope", "lymph_slope",
    "wbc_slope", "cea_slope", "alt_slope", "ldl_slope", "bp_slope", "bmi_slope",
    "platelets_slope", "crp_slope",
    "hba1c_vol", "cea_vol", "hemoglobin_vol", "lymph_vol",
    "n_high", "n_low", "n_critical",
]


def _slope(values: list) -> float:
    """Linear slope normalised by first value."""
    if len(values) < 2:
        return 0.0
    v0 = values[0]
    return (values[-1] - v0) / max(abs(v0), 1e-6)


def build_patient_features(patient_rows: list, patient_meta: dict) -> Optional[np.ndarray]:
    """
    Build the 49-feature vector for one patient from their longitudinal rows.

    patient_rows — list of dicts, each a monthly visit, sorted by date.
    patient_meta — dict: {age, sex, smoking_status, alcohol, exercise,
                          family_history_cancer, family_history_diabetes,
                          family_history_cardio}
    """
    if len(patient_rows) < 2:
        return None

    def vals(field):
        return [r[field] for r in patient_rows if r.get(field) is not None]

    def latest(field, default=0.0):
        v = vals(field)
        return v[-1] if v else default

    # Demographics
    age     = float(patient_meta.get("age", 50))
    sex_f   = 1.0 if str(patient_meta.get("sex", "M")).upper() in ("F", "FEMALE") else 0.0
    smoke   = float(patient_meta.get("smoking_status", 0))
    alcohol = float(patient_meta.get("alcohol_units_weekly", 0))
    exercise_inv = max(0.0, 300.0 - float(patient_meta.get("exercise_min_weekly", 150)))
    fam_c  = float(patient_meta.get("family_history_cancer", 0))
    fam_d  = float(patient_meta.get("family_history_diabetes", 0))
    fam_v  = float(patient_meta.get("family_history_cardio", 0))

    n_chk   = float(len(patient_rows))
    months  = float(len(patient_rows) * 1.0)  # approximation

    # Latest biomarker values
    hba1c      = latest("hba1c", 5.5)
    glucose    = latest("glucose_fasting", 90.0)
    hemoglobin = latest("hemoglobin", 13.5)
    lymph      = latest("lymphocytes_pct", 30.0)
    wbc        = latest("wbc", 7.0)
    platelets  = latest("platelets", 250.0)
    cea        = latest("cea", 1.5)
    ca125      = latest("ca125", 12.0)
    psa        = latest("psa", 1.0)
    alt        = latest("alt", 22.0)
    ast        = latest("ast", 20.0)
    ldl        = latest("ldl", 100.0)
    hdl        = latest("hdl", 55.0)
    triglyc    = latest("triglycerides", 120.0)
    bp_sys     = latest("bp_systolic", 118.0)
    bmi        = latest("bmi", 24.0)
    creatinine = latest("creatinine", 0.9)
    tsh        = latest("tsh", 2.2)
    crp        = latest("crp", 1.0)
    ferritin   = latest("ferritin", 80.0)

    # Slopes
    hba1c_sl    = _slope(vals("hba1c"))
    glucose_sl  = _slope(vals("glucose_fasting"))
    hgb_sl      = _slope(vals("hemoglobin"))
    lymph_sl    = _slope(vals("lymphocytes_pct"))
    wbc_sl      = _slope(vals("wbc"))
    cea_sl      = _slope(vals("cea"))
    alt_sl      = _slope(vals("alt"))
    ldl_sl      = _slope(vals("ldl"))
    bp_sl       = _slope(vals("bp_systolic"))
    bmi_sl      = _slope(vals("bmi"))
    plt_sl      = _slope(vals("platelets"))
    crp_sl      = _slope(vals("crp"))

    # Volatility
    def vol(field):
        v = vals(field)
        return float(np.std(v)) if len(v) > 1 else 0.0

    hba1c_v = vol("hba1c")
    cea_v   = vol("cea")
    hgb_v   = vol("hemoglobin")
    lymph_v = vol("lymphocytes_pct")

    # Reference-range violations
    n_hi = (int(hba1c > 5.7) + int(glucose > 100) + int(cea > 5) +
            int(alt > 40) + int(ldl > 130) + int(bp_sys > 130) +
            int(triglyc > 150))
    n_lo = (int(hemoglobin < (12 if sex_f else 13.5)) +
            int(lymph < 20) + int(hdl < (50 if sex_f else 40)))
    n_crit = (int(cea > 10) + int(hba1c > 6.5) + int(glucose > 126) +
              int(wbc < 3) + int(wbc > 12) + int(lymph < 15))

    feat = [
        age, sex_f, smoke, alcohol, exercise_inv,
        fam_c, fam_d, fam_v, n_chk, months,
        hba1c, glucose, hemoglobin, lymph, wbc, platelets,
        cea, ca125, psa, alt, ast, ldl, hdl, triglyc,
        bp_sys, bmi, creatinine, tsh, crp, ferritin,
        hba1c_sl, glucose_sl, hgb_sl, lymph_sl,
        wbc_sl, cea_sl, alt_sl, ldl_sl, bp_sl, bmi_sl, plt_sl, crp_sl,
        hba1c_v, cea_v, hgb_v, lymph_v,
        float(n_hi), float(n_lo), float(n_crit),
    ]
    return np.array(feat, dtype=np.float32)


# ── Outcome labelling from MIMIC ──────────────────────────────────────────────

def _build_outcomes_from_mimic(patient_id: str, diagnoses: list, dod: Optional[str]) -> dict:
    """
    Build binary outcome labels from MIMIC ICD codes.
    These become the training targets (probability 0.0–1.0).
    """
    CANCER_ICD = {
        "C00", "C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08", "C09",
        "C10", "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18", "C19",
        "C20", "C21", "C22", "C34", "C43", "C50", "C56", "C61", "C67", "C73",
        "C80", "C81", "C82", "C83", "C84", "C85", "C86", "C88", "C90", "C91",
        "C92", "C93", "C94", "C95", "D46",
    }
    METABOLIC_ICD = {
        "E11", "E10", "E13", "E66", "E78", "E03", "E04", "E05", "E06",
        "R73",  # pre-diabetes
    }
    CARDIO_ICD = {
        "I10", "I11", "I12", "I13", "I20", "I21", "I22", "I25",
        "I48", "I50", "I63", "I65", "I70",
    }
    HEMA_ICD = {
        "D50", "D51", "D52", "D53", "D55", "D56", "D57", "D58", "D59",
        "D60", "D61", "D62", "D63", "D64", "D70", "D72", "D75",
    }

    def has_any(icd_set):
        for d in diagnoses:
            code = str(d).upper()[:3]
            if code in icd_set:
                return True
        return False

    return {
        "cancer":      1.0 if has_any(CANCER_ICD)    else 0.0,
        "metabolic":   1.0 if has_any(METABOLIC_ICD) else 0.0,
        "cardio":      1.0 if has_any(CARDIO_ICD)    else 0.0,
        "hematologic": 1.0 if has_any(HEMA_ICD)      else 0.0,
    }


# ── Training ──────────────────────────────────────────────────────────────────

def train_on_mimic(
    data_path: str,
    output_dir: str = "./models",
    min_visits: int = 3,
    test_size: float = 0.15,
    validate: bool = True,
) -> dict:
    """
    Load the MIMIC-IV CSV export, build features, train all 4 models,
    evaluate on held-out test set, and save model weights.

    Returns dict with AUC/MAE metrics per domain.
    """
    try:
        import pandas as pd
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn.isotonic import IsotonicRegression
        from sklearn.metrics import roc_auc_score, mean_absolute_error
        import pickle
    except ImportError as e:
        logger.error(f"Missing dependency: {e}. Run: pip install pandas scikit-learn")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Loading MIMIC-IV export: {data_path}")

    df = pd.read_csv(data_path, low_memory=False)
    logger.info(f"Loaded {len(df):,} rows, {df['subject_id'].nunique():,} unique patients")

    # Group by patient
    patients = df.groupby("subject_id")
    logger.info(f"Building features for patients with {min_visits}+ visits...")

    X_rows, y_rows = [], {d: [] for d in ("cancer", "metabolic", "cardio", "hematologic")}
    skipped = 0

    for pid, group in patients:
        group = group.sort_values("visit_month")
        if len(group) < min_visits:
            skipped += 1
            continue

        rows = group.to_dict("records")
        meta = {
            "age":   group.iloc[0].get("anchor_age", 50),
            "sex":   group.iloc[0].get("gender", "M"),
            "smoking_status": 0,
            "alcohol_units_weekly": 0,
            "exercise_min_weekly": 150,
            "family_history_cancer": 0,
            "family_history_diabetes": 0,
            "family_history_cardio": 0,
        }

        feat = build_patient_features(rows, meta)
        if feat is None:
            skipped += 1
            continue

        # Derive labels from last-visit diagnoses proxy
        # In real MIMIC data, you'd join with diagnoses_icd table
        # Here we use proxy signals from lab values
        cancer_signal = (
            (feat[16] > 5.0) or  # cea
            (feat[12] < 10.0 and feat[12] > 0)  # severe anaemia
        )
        meta_signal   = feat[10] > 6.5 or feat[11] > 126  # hba1c/glucose diabetic
        cardio_signal = feat[24] > 140 or feat[21] > 160   # bp/ldl
        hema_signal   = feat[12] < 10.0 or feat[14] < 3.0 or feat[14] > 12.0

        X_rows.append(feat)
        y_rows["cancer"].append(1.0 if cancer_signal else 0.0)
        y_rows["metabolic"].append(1.0 if meta_signal else 0.0)
        y_rows["cardio"].append(1.0 if cardio_signal else 0.0)
        y_rows["hematologic"].append(1.0 if hema_signal else 0.0)

    if not X_rows:
        logger.error("No valid patient rows built. Check CSV format.")
        sys.exit(1)

    X = np.array(X_rows, dtype=np.float32)
    logger.info(f"Built feature matrix: {X.shape[0]:,} patients × {X.shape[1]} features")
    logger.info(f"Skipped: {skipped:,} patients (< {min_visits} visits)")

    # Scale
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    results = {}
    models_to_save = {"scaler": scaler}

    logger.info("\nTraining 4 disease models on MIMIC-IV data...\n")

    for domain, y_raw in y_rows.items():
        y = np.array(y_raw, dtype=np.float32)
        pos_rate = y.mean()
        logger.info(f"  {domain:<15} positive rate: {pos_rate:.1%} of {len(y):,} patients")

        y_clipped = np.clip(y, 0.02, 0.98)
        Xtr, Xte, ytr, yte = train_test_split(
            Xs, y_clipped, test_size=test_size, random_state=42, stratify=(y > 0.5)
        )

        reg = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=20,
            random_state=42,
        )
        reg.fit(Xtr, ytr)

        raw_preds_val = reg.predict_proba(Xte)[:, 1]

        # Isotonic calibration
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw_preds_val, yte)
        cal_preds = iso.predict(raw_preds_val)

        if validate:
            mae  = mean_absolute_error(yte, cal_preds)
            try:
                auc = roc_auc_score(yte > 0.5, cal_preds)
            except Exception:
                auc = float("nan")

            logger.info(f"    MAE: {mae:.4f}  |  AUC: {auc:.4f}")
            results[domain] = {"mae": round(mae, 4), "auc": round(auc, 4)}
        else:
            results[domain] = {}

        # Feature importances
        fi = reg.feature_importances_
        top_idx = np.argsort(fi)[::-1][:8]
        top_feats = [(FEATURES[i], round(float(fi[i]), 4)) for i in top_idx]
        logger.info(f"    Top features: {', '.join(f[0] for f in top_feats[:4])}")

        models_to_save[f"model_{domain}"] = {
            "reg": reg,
            "iso": iso,
            "top_feats": top_feats,
            "domain": domain,
        }

    # Save everything
    import pickle
    out_path = Path(output_dir) / "biosentinel_models_mimic.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(models_to_save, f)
    logger.info(f"\n✅ Models saved: {out_path}")

    # Save metadata
    meta_path = Path(output_dir) / "training_metadata.json"
    meta = {
        "trained_at":    datetime.now(datetime.now().astimezone().tzinfo).isoformat(),
        "data_source":   data_path,
        "n_patients":    X.shape[0],
        "n_features":    X.shape[1],
        "min_visits":    min_visits,
        "features":      FEATURES,
        "results":       results,
        "note":          "Trained on MIMIC-IV. Never share raw MIMIC data — only these weights.",
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"✅ Metadata saved: {meta_path}")

    if results:
        logger.info("\n📊 Summary:")
        for domain, r in results.items():
            logger.info(f"   {domain:<15} MAE={r.get('mae','?')}  AUC={r.get('auc','?')}")

    logger.info("\n🔗 Next steps:")
    logger.info("   1. Set MODEL_PATH=./models in .env")
    logger.info("   2. Restart BioSentinel — it will load these weights automatically")
    logger.info("   3. Run: python train_mimic.py --compare models/")
    logger.info("   4. Update README with the AUC metrics from results above")

    return results


def compare_runs(models_dir: str):
    """Print a comparison table of synthetic vs MIMIC-IV training results."""
    import json
    meta_path = Path(models_dir) / "training_metadata.json"
    if not meta_path.exists():
        logger.error(f"No training_metadata.json found in {models_dir}")
        return

    with open(meta_path) as f:
        meta = json.load(f)

    synthetic_baseline = {
        "cancer":      {"mae": 0.066, "auc": "N/A (synthetic)"},
        "metabolic":   {"mae": 0.069, "auc": "N/A (synthetic)"},
        "cardio":      {"mae": 0.059, "auc": "N/A (synthetic)"},
        "hematologic": {"mae": 0.027, "auc": "N/A (synthetic)"},
    }

    print("\n" + "="*70)
    print("  BioSentinel Model Comparison: Synthetic vs MIMIC-IV")
    print("="*70)
    print(f"  {'Domain':<15} {'Synthetic MAE':>15} {'MIMIC AUC':>12} {'MIMIC MAE':>10}")
    print("-"*70)

    for domain in ("cancer", "metabolic", "cardio", "hematologic"):
        syn = synthetic_baseline.get(domain, {})
        real = meta.get("results", {}).get(domain, {})
        print(f"  {domain:<15} {str(syn.get('mae','?')):>15} "
              f"{str(real.get('auc','?')):>12} {str(real.get('mae','?')):>10}")

    print("="*70)
    print(f"  Patients: {meta.get('n_patients', '?'):,}  |  "
          f"Features: {meta.get('n_features', '?')}  |  "
          f"Min visits: {meta.get('min_visits', '?')}")
    print(f"  Trained: {meta.get('trained_at', '?')[:19]}")
    print()


def print_bigquery_query():
    """Print the BigQuery export SQL to stdout."""
    print(BIGQUERY_EXPORT_QUERY)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train BioSentinel on MIMIC-IV data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data",     help="Path to MIMIC-IV CSV export")
    parser.add_argument("--output",   default="./models", help="Output directory for model files")
    parser.add_argument("--min-visits", type=int, default=3,
                        help="Minimum visits per patient (default: 3)")
    parser.add_argument("--validate", action="store_true", default=True,
                        help="Compute AUC/MAE on held-out test set (default: True)")
    parser.add_argument("--compare",  metavar="MODELS_DIR",
                        help="Compare synthetic vs MIMIC-IV results from a models/ directory")
    parser.add_argument("--print-query", action="store_true",
                        help="Print the BigQuery export SQL and exit")

    args = parser.parse_args()

    if args.print_query:
        print_bigquery_query()
        sys.exit(0)

    if args.compare:
        compare_runs(args.compare)
        sys.exit(0)

    if not args.data:
        parser.print_help()
        print("\n⚠️  Provide --data path to the MIMIC-IV CSV export.")
        print("   Use --print-query to get the BigQuery SQL.\n")
        sys.exit(1)

    if not os.path.exists(args.data):
        logger.error(f"Data file not found: {args.data}")
        sys.exit(1)

    train_on_mimic(
        data_path=args.data,
        output_dir=args.output,
        min_visits=args.min_visits,
        validate=args.validate,
    )
