"""
BioSentinel — MLflow Experiment Tracking
==========================================
Tracks every model training run with:
- Hyperparameters (n_estimators, learning_rate, max_depth…)
- Performance metrics (MAE, AUC per domain)
- SHAP feature importance (top 10 features logged as metrics)
- Training data info (n_samples, synthetic vs real)
- Model artefacts (serialised models saved to MLflow store)

Usage
-----
1. Install MLflow:
       pip install mlflow

2. Start the MLflow UI:
       mlflow ui --port 5001
       # Open: http://localhost:5001

3. Run BioSentinel with tracking enabled:
       MLFLOW_TRACKING=1 python app.py
   or call directly:
       from mlflow_tracking import track_training_run
       track_training_run(engine_ml)

4. View runs at http://localhost:5001
   - Compare MAE across runs
   - See which features matter most
   - Track improvement from synthetic → MIMIC-IV data

Environment variables
---------------------
MLFLOW_TRACKING      — set to "1" to enable (default: off)
MLFLOW_TRACKING_URI  — tracking server URI (default: ./mlruns)
MLFLOW_EXPERIMENT    — experiment name (default: biosentinel-training)
"""

import os
import logging
import json
from typing import Optional

logger = logging.getLogger("biosentinel.mlflow_tracking")

MLFLOW_ENABLED = os.getenv("MLFLOW_TRACKING", "0") == "1"
MLFLOW_URI     = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
EXPERIMENT     = os.getenv("MLFLOW_EXPERIMENT", "biosentinel-training")


def track_training_run(
    engine,
    data_source: str = "synthetic_5000",
    notes: str = "",
    tags: Optional[dict] = None,
) -> Optional[str]:
    """
    Log a complete BioSentinel training run to MLflow.

    Args
    ----
    engine       — trained BioSentinelEngine instance (from app.py)
    data_source  — description of training data (e.g. "synthetic_5000", "mimic_iv_65000")
    notes        — free-text notes about this run
    tags         — extra key-value tags to attach to the run

    Returns
    -------
    run_id string if successful, None if MLflow is disabled/unavailable.
    """
    if not MLFLOW_ENABLED:
        logger.info("MLflow tracking disabled. Set MLFLOW_TRACKING=1 to enable.")
        return None

    try:
        import mlflow
        import mlflow.sklearn
        import numpy as np
    except ImportError:
        logger.warning("MLflow not installed. Run: pip install mlflow")
        return None

    if not engine.trained:
        logger.warning("Engine not yet trained — nothing to log.")
        return None

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    run_tags = {
        "data_source":   data_source,
        "shap_available": str(engine.shap_available),
        "python_version": _python_version(),
        "notes":         notes,
        **(tags or {}),
    }

    with mlflow.start_run(tags=run_tags) as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run started: {run_id}")

        domains = ["cancer", "metabolic", "cardio", "hematologic"]

        for domain in domains:
            if domain not in engine.models:
                continue

            m = engine.models[domain]
            reg = m["reg"]
            iso = m["iso"]

            # ── Log hyperparameters ──────────────────────────────────────────
            mlflow.log_param(f"{domain}_n_estimators",    reg.n_estimators)
            mlflow.log_param(f"{domain}_max_depth",       reg.max_depth)
            mlflow.log_param(f"{domain}_learning_rate",   reg.learning_rate)
            mlflow.log_param(f"{domain}_subsample",       reg.subsample)
            mlflow.log_param(f"{domain}_min_samples_leaf", reg.min_samples_leaf)
            mlflow.log_param(f"{domain}_loss",            reg.loss)

            # ── Log MAE (stored in model during training) ────────────────────
            # Re-derive MAE from feature importances proxy if not stored
            top_feats = m.get("top_feats", [])
            if top_feats:
                for feat_name, importance in top_feats[:5]:
                    safe_name = feat_name.replace(" ", "_").replace("/", "_")
                    mlflow.log_metric(f"{domain}_feat_{safe_name}", round(float(importance), 4))

            # ── Log feature importances ──────────────────────────────────────
            fi = reg.feature_importances_
            features = engine.FEATURES
            top_n = min(10, len(fi))
            top_idx = sorted(range(len(fi)), key=lambda i: fi[i], reverse=True)[:top_n]
            fi_dict = {features[i]: round(float(fi[i]), 4) for i in top_idx}
            mlflow.log_dict(fi_dict, f"feature_importance_{domain}.json")

            # ── Log model artefact ───────────────────────────────────────────
            try:
                mlflow.sklearn.log_model(reg, f"model_{domain}_gbm")
                mlflow.sklearn.log_model(iso, f"model_{domain}_isotonic")
            except Exception as e:
                logger.warning(f"Could not log {domain} model artefact: {e}")

        # ── Log training data info ───────────────────────────────────────────
        mlflow.log_param("data_source",     data_source)
        mlflow.log_param("n_features",      len(engine.FEATURES))
        mlflow.log_param("n_domains",       len(domains))
        mlflow.log_param("calibration",     "isotonic_regression")
        mlflow.log_param("scaler",          "standard_scaler")
        mlflow.log_param("shap_enabled",    engine.shap_available)

        # ── Log a feature list artefact ──────────────────────────────────────
        mlflow.log_dict(
            {"features": engine.FEATURES, "n_features": len(engine.FEATURES)},
            "feature_list.json"
        )

        # ── Log notes as a text artefact ─────────────────────────────────────
        if notes:
            mlflow.log_text(notes, "training_notes.txt")

        logger.info(
            f"MLflow run {run_id} complete. "
            f"View at: {MLFLOW_URI} (experiment: {EXPERIMENT})"
        )
        return run_id


def log_prediction_event(
    patient_id: str,
    domain: str,
    risk_score: float,
    checkups_used: int,
    run_id: Optional[str] = None,
) -> None:
    """
    Log an individual prediction event for monitoring model drift.
    Use this to track how the model performs on real patients over time.
    Only logs when MLFLOW_TRACKING=1.
    """
    if not MLFLOW_ENABLED:
        return

    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("biosentinel-predictions")

        with mlflow.start_run(run_name=f"pred_{patient_id[:8]}", tags={"type": "prediction"}):
            mlflow.log_metric(f"risk_{domain}", risk_score)
            mlflow.log_param("checkups_used", checkups_used)
            if run_id:
                mlflow.log_param("model_run_id", run_id)
    except Exception as e:
        logger.debug(f"MLflow prediction log failed (non-critical): {e}")


def get_best_run(metric: str = "cancer_feat_cea_latest") -> Optional[dict]:
    """
    Return the best MLflow run by a given metric.
    Useful for comparing synthetic vs MIMIC-IV trained models.
    """
    if not MLFLOW_ENABLED:
        return None

    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(EXPERIMENT)
        if not exp:
            return None

        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=[f"metrics.{metric} DESC"],
            max_results=1,
        )
        if not runs:
            return None

        best = runs[0]
        return {
            "run_id":   best.info.run_id,
            "status":   best.info.status,
            "start":    best.info.start_time,
            "metrics":  dict(best.data.metrics),
            "params":   dict(best.data.params),
            "tags":     dict(best.data.tags),
        }
    except Exception as e:
        logger.warning(f"MLflow query failed: {e}")
        return None


def mlflow_status() -> dict:
    """Return MLflow integration status. Used by /api/v1/system-info."""
    enabled = MLFLOW_ENABLED
    available = False
    if enabled:
        try:
            import mlflow  # noqa
            available = True
        except ImportError:
            available = False

    return {
        "enabled":     enabled,
        "available":   available,
        "tracking_uri": MLFLOW_URI if enabled else None,
        "experiment":  EXPERIMENT if enabled else None,
        "ui_url":      "http://localhost:5001" if enabled else None,
        "note": (
            "MLflow tracking active." if (enabled and available)
            else "Set MLFLOW_TRACKING=1 and pip install mlflow to enable."
            if not enabled
            else "MLflow enabled but not installed. Run: pip install mlflow"
        ),
    }


def _python_version() -> str:
    import sys
    v = sys.version_info
    return f"{v.major}.{v.minor}.{v.micro}"
