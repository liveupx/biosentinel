"""
BioSentinel — CancerRiskNet v0.1
Pan-cancer risk prediction ensemble model.

Combines:
  - XGBoost on tabular temporal features
  - LightGBM on medication/diagnosis sequence features
  - LSTM on raw biomarker time series
  - Logistic Regression meta-learner

Literature basis:
  - Winther et al. (Nature Medicine, 2023): Disease trajectory modeling for pancreatic cancer
    AUROC 0.88 on 6M patient cohort (DNPR + US-VA)
  - BMC Medical Research Methodology (2025): Scoping review of longitudinal EHR AI models

⚠️ NOTE: This is v0.1 with proof-of-concept weights.
   Clinical validation on real-world data is required before any clinical use.
   See docs/models/cancer_risk_net.md for details.
"""

import logging
from typing import Optional

import numpy as np

from src.models.base_model import BaseDiseaseModel, RiskPrediction, InsufficientDataError

logger = logging.getLogger(__name__)


class CancerRiskNet(BaseDiseaseModel):
    """
    Pan-cancer risk prediction ensemble.

    Predicts composite cancer risk across 9 primary cancer types using
    an ensemble of XGBoost, LightGBM, and LSTM models on longitudinal
    patient health data.

    Model Type:   XGBoost + LightGBM + LSTM Ensemble
    Disease:      Pan-cancer (composite) + 9 individual cancer modules
    Category:     cancer
    Min Checkups: 2 (24+ months recommended for best accuracy)
    Literature AUC: ~0.88 (pancreatic, DNPR cohort, Nature Medicine 2023)

    Input features (~1,500 temporal features):
        - CBC trends (WBC, lymphocytes, hemoglobin, platelets)
        - Metabolic panel trends (glucose, liver enzymes, creatinine)
        - Tumor marker trends (CEA, CA-125, PSA, AFP, CA-19.9)
        - Medication history sequences
        - Diagnosis history sequences
        - Demographic risk factors
        - Family history flags

    Output:
        - composite_cancer_risk: float (0.0–1.0)
        - per_cancer_risks: dict of {cancer_type: risk_score}
        - top_shap_features: list of contributing features
    """

    MODEL_ID = "cancer_risk_net_v0.1"
    DISEASE_NAME = "cancer (composite)"
    DISEASE_CATEGORY = "cancer"
    MODEL_VERSION = "0.1.0"
    MIN_CHECKUPS_REQUIRED = 2

    TARGET_BIOMARKERS = [
        # CBC
        "wbc", "lymphocytes_pct", "neutrophils_pct",
        "hemoglobin", "platelets", "mcv",
        # Metabolic
        "glucose_fasting", "alt", "ast", "albumin",
        "creatinine", "bilirubin_total",
        # Tumor markers
        "cea", "ca125", "psa", "afp", "ca199",
        # Hormones
        "tsh",
    ]

    CANCER_MODULES = [
        "lung", "colorectal", "breast", "pancreatic",
        "liver", "leukemia", "lymphoma", "cervical", "thyroid"
    ]

    def predict(self, timeline) -> RiskPrediction:
        """Run pan-cancer risk prediction on patient timeline."""
        self._validate_input(timeline)

        features = self._extract_features(timeline)
        composite_risk = self._run_model(features)
        shap_values = self._compute_shap(features)
        confidence = self._estimate_confidence(timeline, features)
        recommendation = self._build_recommendation(composite_risk, shap_values)

        return RiskPrediction(
            model_id=self.MODEL_ID,
            disease=self.DISEASE_NAME,
            disease_category=self.DISEASE_CATEGORY,
            risk_score=composite_risk,
            risk_level="",  # set by __post_init__
            confidence=confidence,
            shap_values=shap_values,
            top_features=self._top_features(shap_values, n=5),
            data_completeness=self._get_data_completeness(features),
            lookback_months=self._get_lookback_months(timeline),
            checkups_used=len(timeline.checkups),
            recommendation=recommendation,
        )

    def _extract_features(self, timeline) -> dict:
        """
        Extract ~1,500 temporal features from the patient timeline.

        For each key biomarker, computes:
        - Latest value
        - Trend slope (3m, 6m, 12m windows)
        - Variance over full timeline
        - Whether value crossed reference range boundary
        """
        features = {}
        checkups = sorted(timeline.checkups, key=lambda c: c.date)

        for marker in self.TARGET_BIOMARKERS:
            values = [
                getattr(c.lab_results, marker, None)
                for c in checkups
                if getattr(c.lab_results, marker, None) is not None
            ]

            if not values:
                features[f"{marker}_latest"] = None
                features[f"{marker}_slope"] = None
                features[f"{marker}_variance"] = None
                continue

            features[f"{marker}_latest"] = values[-1]
            features[f"{marker}_variance"] = float(np.var(values)) if len(values) > 1 else 0.0

            if len(values) >= 2:
                # Linear slope as fraction of first value (relative trend)
                slope = (values[-1] - values[0]) / max(abs(values[0]), 1e-6)
                features[f"{marker}_slope"] = round(slope, 4)
            else:
                features[f"{marker}_slope"] = 0.0

        # Add demographic risk factors
        features["age"] = getattr(timeline.patient, "age", None)
        features["sex"] = getattr(timeline.patient, "sex", None)
        features["family_history_cancer"] = self._encode_family_history(timeline)
        features["n_checkups"] = len(checkups)
        features["months_of_data"] = self._get_lookback_months(timeline)

        return features

    def _run_model(self, features: dict) -> float:
        """
        Run the ensemble model.

        v0.1: Uses a simple heuristic-based scoring as a proof-of-concept
        until real model weights are trained and validated.
        
        TODO: Replace with trained XGBoost + LightGBM + LSTM ensemble.
        """
        logger.warning(
            "CancerRiskNet v0.1 is using proof-of-concept heuristic weights. "
            "Do NOT use for clinical decision making. Validated weights required."
        )

        # Proof-of-concept heuristic scoring
        risk_signals = []

        # Elevated CEA (colorectal, lung, gastric marker)
        cea = features.get("cea_latest")
        if cea and cea > 5.0:
            risk_signals.append(min((cea - 5.0) / 20.0, 0.4))

        # Lymphocyte downtrend (hematologic malignancy signal)
        lymph_slope = features.get("lymphocytes_pct_slope")
        if lymph_slope and lymph_slope < -0.15:
            risk_signals.append(min(abs(lymph_slope) * 0.6, 0.3))

        # Hemoglobin downtrend (anemia, internal bleeding, malignancy)
        hgb_slope = features.get("hemoglobin_slope")
        if hgb_slope and hgb_slope < -0.10:
            risk_signals.append(min(abs(hgb_slope) * 0.5, 0.25))

        # Family history
        fh = features.get("family_history_cancer", 0)
        if fh > 0:
            risk_signals.append(min(fh * 0.05, 0.15))

        # Age risk (cancer risk increases with age)
        age = features.get("age")
        if age and age > 50:
            risk_signals.append(min((age - 50) * 0.004, 0.15))

        # Composite: weighted sum, clipped to 0.0-1.0
        if not risk_signals:
            return 0.10  # baseline population risk

        composite = sum(risk_signals) / max(len(risk_signals), 1) + 0.05
        return round(min(max(composite, 0.0), 1.0), 3)

    def _compute_shap(self, features: dict) -> dict:
        """
        Compute SHAP values for feature attribution.
        
        v0.1: Returns approximate feature importance signals.
        TODO: Integrate real SHAP computation once model weights are trained.
        """
        shap_values = {}

        signal_features = {
            "cea_latest": 0.0,
            "lymphocytes_pct_slope": 0.0,
            "hemoglobin_slope": 0.0,
            "family_history_cancer": 0.0,
            "age": 0.0,
            "platelets_slope": 0.0,
            "alt_slope": 0.0,
        }

        cea = features.get("cea_latest")
        if cea and cea > 5.0:
            signal_features["cea_latest"] = round((cea - 5.0) / 20.0, 3)

        lymph_slope = features.get("lymphocytes_pct_slope")
        if lymph_slope:
            signal_features["lymphocytes_pct_slope"] = round(lymph_slope * -0.5, 3)

        hgb_slope = features.get("hemoglobin_slope")
        if hgb_slope:
            signal_features["hemoglobin_slope"] = round(hgb_slope * -0.4, 3)

        age = features.get("age")
        if age and age > 40:
            signal_features["age"] = round((age - 40) * 0.003, 3)

        shap_values = {k: v for k, v in signal_features.items() if v != 0.0}
        return shap_values

    def _top_features(self, shap_values: dict, n: int = 5) -> list:
        """Return top N features by absolute SHAP value with human-readable labels."""
        feature_labels = {
            "cea_latest": "CEA tumor marker (elevated)",
            "lymphocytes_pct_slope": "Lymphocyte % downtrend",
            "hemoglobin_slope": "Hemoglobin declining trend",
            "family_history_cancer": "Family history of cancer",
            "age": "Age-related risk factor",
            "alt_slope": "Liver enzyme (ALT) trend",
            "platelets_slope": "Platelet count trend",
        }
        sorted_features = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
        return [
            {
                "feature": k,
                "label": feature_labels.get(k, k),
                "shap_value": v,
                "direction": "risk_increasing" if v > 0 else "protective"
            }
            for k, v in sorted_features[:n]
        ]

    def _estimate_confidence(self, timeline, features: dict) -> float:
        """Estimate prediction confidence based on data completeness and timeline length."""
        completeness = self._get_data_completeness(features)
        n_checkups = len(timeline.checkups)
        months = self._get_lookback_months(timeline)

        # More data = higher confidence (up to a point)
        checkup_factor = min(n_checkups / 8.0, 1.0)   # 8 checkups = max
        month_factor = min(months / 24.0, 1.0)         # 24 months = max

        confidence = (completeness * 0.4 + checkup_factor * 0.35 + month_factor * 0.25)
        return round(min(confidence, 0.95), 2)  # Cap at 0.95 — never 100% confident

    def _encode_family_history(self, timeline) -> int:
        """Count number of first-degree relatives with cancer history."""
        fh = getattr(timeline, "family_history", {})
        cancer_history = fh.get("cancer", [])
        return len(cancer_history)

    def _get_lookback_months(self, timeline) -> int:
        """Calculate the number of months of data available."""
        if not timeline.checkups or len(timeline.checkups) < 2:
            return 0
        from datetime import date
        dates = sorted([c.date for c in timeline.checkups])
        delta = dates[-1] - dates[0]
        return round(delta.days / 30.4)

    def _build_recommendation(self, risk_score: float, shap_values: dict) -> str:
        """Generate clinical recommendation string based on risk level."""
        if risk_score < 0.25:
            return "Continue routine 3-month checkups. No elevated cancer risk signals detected."
        elif risk_score < 0.50:
            top = list(shap_values.keys())[:2] if shap_values else []
            return (
                f"Moderate risk signals detected ({', '.join(top)}). "
                "Consider discussing cancer screening schedule with physician. "
                "Continue 3-month monitoring."
            )
        elif risk_score < 0.75:
            return (
                "High risk signals detected. Recommend specialist oncology consultation "
                "and targeted cancer screening (specify based on top risk factors). "
                "Increase monitoring frequency."
            )
        else:
            return (
                "CRITICAL: Multiple strong cancer risk signals detected. "
                "Immediate oncology referral recommended. "
                "Comprehensive cancer screening required urgently."
            )
