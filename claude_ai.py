"""
BioSentinel — Claude AI Integration Module
==========================================
Provides three AI-powered capabilities:

1. Lab Report Vision OCR
   Sends a lab report image/PDF page to Claude Vision and extracts
   biomarker values as structured JSON. Used as a high-accuracy
   fallback when pdfplumber / pytesseract fail or produce garbage.

2. Plain-English Narrative Generation
   After every ML prediction, calls Claude Haiku to write a 3–4
   sentence summary the patient (not just the clinician) can understand.
   Includes the top risk drivers in everyday language.

3. Longitudinal Trend Anomaly Narration
   Passes the full biomarker timeline as JSON to Claude Sonnet and
   asks for unusual patterns that fixed numeric thresholds miss —
   e.g. CEA creeping up for 18 months while still "within limits".

Usage
-----
Set the ANTHROPIC_API_KEY environment variable. All functions degrade
gracefully if the key is missing — they return None and log a warning.

  export ANTHROPIC_API_KEY="sk-ant-..."

Then use:
  from claude_ai import (
      extract_labs_from_image,
      generate_prediction_narrative,
      detect_trend_anomalies,
  )

IMPORTANT — MIMIC-IV restriction
---------------------------------
This module must NEVER be used with real MIMIC-IV patient data.
The PhysioNet DUA prohibits sending credentialed data to external APIs.
Use only with your own clinic's data where you hold appropriate consent.
"""

import base64
import io
import json
import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger("biosentinel.claude_ai")

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
SONNET_MODEL = "claude-sonnet-4-20250514"
HAIKU_MODEL  = "claude-haiku-4-5-20251001"

# ── Reference ranges for context injection ────────────────────────────────────
BIOMARKER_REFS = {
    "hba1c":            {"unit": "%",      "normal": "4.0–5.6",  "prediab": "5.7–6.4", "diab": ">6.5"},
    "glucose_fasting":  {"unit": "mg/dL",  "normal": "70–99",    "prediab": "100–125",  "diab": ">126"},
    "hemoglobin":       {"unit": "g/dL",   "normal": "12–17.5 (sex-dep)"},
    "lymphocytes_pct":  {"unit": "%",      "normal": "20–40"},
    "wbc":              {"unit": "K/µL",   "normal": "4.5–11.0"},
    "cea":              {"unit": "ng/mL",  "normal": "<3.0 (non-smoker)"},
    "ca125":            {"unit": "U/mL",   "normal": "<35"},
    "psa":              {"unit": "ng/mL",  "normal": "<4.0 (age-dep)"},
    "alt":              {"unit": "U/L",    "normal": "7–56"},
    "ldl":              {"unit": "mg/dL",  "normal": "<100 optimal"},
    "hdl":              {"unit": "mg/dL",  "normal": ">60 optimal"},
    "bp_systolic":      {"unit": "mmHg",   "normal": "<120"},
    "bmi":              {"unit": "kg/m²",  "normal": "18.5–24.9"},
    "crp":              {"unit": "mg/L",   "normal": "<1.0 low risk"},
    "creatinine":       {"unit": "mg/dL",  "normal": "0.6–1.2"},
    "tsh":              {"unit": "mIU/L",  "normal": "0.4–4.0"},
    "ferritin":         {"unit": "ng/mL",  "normal": "12–300 (sex-dep)"},
    "triglycerides":    {"unit": "mg/dL",  "normal": "<150"},
    "platelets":        {"unit": "K/µL",   "normal": "150–400"},
}


def _api_key() -> Optional[str]:
    return os.getenv("ANTHROPIC_API_KEY")


def _call_claude(model: str, messages: list, max_tokens: int = 1000,
                 system: Optional[str] = None) -> Optional[str]:
    """
    Internal helper — makes a single call to the Anthropic Messages API.
    Returns the text content of the first response block, or None on error.
    """
    key = _api_key()
    if not key:
        logger.warning(
            "ANTHROPIC_API_KEY not set — Claude AI features disabled. "
            "Set the env var to enable narrative generation and vision OCR."
        )
        return None

    headers = {
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body: dict = {"model": model, "max_tokens": max_tokens, "messages": messages}
    if system:
        body["system"] = system

    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(ANTHROPIC_API_URL, headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()
            blocks = data.get("content", [])
            for block in blocks:
                if block.get("type") == "text":
                    return block["text"]
            return None
    except httpx.HTTPStatusError as e:
        logger.error(f"Claude API HTTP error: {e.response.status_code} — {e.response.text[:300]}")
        return None
    except Exception as e:
        logger.error(f"Claude API call failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 1.  LAB REPORT VISION OCR
# ══════════════════════════════════════════════════════════════════════════════

def extract_labs_from_image(
    image_bytes: bytes,
    media_type: str = "image/jpeg",
) -> dict:
    """
    Send a lab report image to Claude Vision and extract biomarker values.

    Args
    ----
    image_bytes  Raw image bytes (JPEG, PNG, WEBP, GIF supported by Claude).
    media_type   MIME type — "image/jpeg", "image/png", "image/webp", "image/gif".

    Returns
    -------
    dict with keys:
      "values"  — dict of {biomarker_name: float}  (only those found)
      "method"  — "claude_vision"
      "model"   — model name used
      "error"   — str, only if call failed
    """
    if not _api_key():
        return {"error": "ANTHROPIC_API_KEY not set", "values": {}}

    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    system_prompt = (
        "You are a medical lab report data extraction system. "
        "Your ONLY job is to extract numeric biomarker values from lab reports. "
        "You MUST respond with valid JSON only — no prose, no markdown, no explanation. "
        "Never invent values. If a value is not visible, omit the field entirely."
    )

    user_prompt = (
        "Extract all biomarker values from this lab report image. "
        "Return ONLY a JSON object with this exact structure:\n\n"
        "{\n"
        "  \"hba1c\": <float or null>,\n"
        "  \"glucose_fasting\": <float or null>,\n"
        "  \"hemoglobin\": <float or null>,\n"
        "  \"hematocrit\": <float or null>,\n"
        "  \"wbc\": <float or null>,\n"
        "  \"platelets\": <float or null>,\n"
        "  \"lymphocytes_pct\": <float or null>,\n"
        "  \"neutrophils_pct\": <float or null>,\n"
        "  \"cea\": <float or null>,\n"
        "  \"ca125\": <float or null>,\n"
        "  \"psa\": <float or null>,\n"
        "  \"afp\": <float or null>,\n"
        "  \"alt\": <float or null>,\n"
        "  \"ast\": <float or null>,\n"
        "  \"ldl\": <float or null>,\n"
        "  \"hdl\": <float or null>,\n"
        "  \"total_cholesterol\": <float or null>,\n"
        "  \"triglycerides\": <float or null>,\n"
        "  \"creatinine\": <float or null>,\n"
        "  \"urea\": <float or null>,\n"
        "  \"uric_acid\": <float or null>,\n"
        "  \"tsh\": <float or null>,\n"
        "  \"t3\": <float or null>,\n"
        "  \"t4\": <float or null>,\n"
        "  \"crp\": <float or null>,\n"
        "  \"ferritin\": <float or null>,\n"
        "  \"vitamin_d\": <float or null>,\n"
        "  \"vitamin_b12\": <float or null>,\n"
        "  \"bp_systolic\": <float or null>,\n"
        "  \"bp_diastolic\": <float or null>,\n"
        "  \"bmi\": <float or null>,\n"
        "  \"weight_kg\": <float or null>,\n"
        "  \"albumin\": <float or null>,\n"
        "  \"bilirubin_total\": <float or null>,\n"
        "  \"notes\": \"any relevant notes about the report or patient (optional)\"\n"
        "}\n\n"
        "Rules:\n"
        "- Omit fields where no value is present (don't include null fields).\n"
        "- Convert all units to the standard units above if needed "
        "  (e.g. mmol/L HbA1c → % by: value × 10.929 if IFCC, otherwise check context).\n"
        "- If the report shows HbA1c in mmol/mol (IFCC), convert: % = (mmol_mol / 10.929) + 2.15\n"
        "- If glucose is in mmol/L, convert to mg/dL: value × 18.016\n"
        "- Extract the patient name, date, and lab name into the notes field.\n"
        "- ONLY output the JSON. No surrounding text."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": b64,
                    },
                },
                {"type": "text", "text": user_prompt},
            ],
        }
    ]

    raw = _call_claude(SONNET_MODEL, messages, max_tokens=800, system=system_prompt)
    if raw is None:
        return {"error": "Claude Vision call failed", "values": {}}

    # Strip markdown fences if Claude added them despite instructions
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw

    try:
        parsed = json.loads(raw)
        # Keep only numeric values; discard null and string fields except notes
        values = {}
        notes = parsed.pop("notes", None)
        for k, v in parsed.items():
            if isinstance(v, (int, float)) and v is not None:
                values[k] = float(v)
        result = {"values": values, "method": "claude_vision", "model": SONNET_MODEL}
        if notes:
            result["notes"] = notes
        return result
    except json.JSONDecodeError as e:
        logger.error(f"Claude vision JSON parse error: {e}\nRaw: {raw[:500]}")
        return {"error": f"JSON parse failed: {e}", "values": {}, "raw": raw[:500]}


def extract_labs_from_pdf_pages(pdf_bytes: bytes) -> dict:
    """
    Extract labs from a PDF by converting the first 3 pages to images
    and passing them to Claude Vision. Requires the 'pdf2image' package
    and Poppler installed on the system.

    Fallback: if pdf2image is not available, returns an error dict.
    """
    try:
        from pdf2image import convert_from_bytes
    except ImportError:
        return {
            "error": "pdf2image not installed. Run: pip install pdf2image",
            "values": {},
        }

    try:
        pages = convert_from_bytes(pdf_bytes, first_page=1, last_page=3, dpi=200)
    except Exception as e:
        return {"error": f"PDF conversion failed: {e}", "values": {}}

    merged_values = {}
    for page_img in pages:
        buf = io.BytesIO()
        page_img.save(buf, format="JPEG", quality=90)
        result = extract_labs_from_image(buf.getvalue(), media_type="image/jpeg")
        if "values" in result:
            # Later pages overwrite earlier ones for the same field
            merged_values.update(result["values"])

    return {"values": merged_values, "method": "claude_vision_pdf",
            "model": SONNET_MODEL, "pages_processed": len(pages)}


# ══════════════════════════════════════════════════════════════════════════════
# 2.  PLAIN-ENGLISH NARRATIVE GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_prediction_narrative(
    prediction: dict,
    patient_info: dict,
    audience: str = "patient",
) -> Optional[str]:
    """
    Generate a plain-English summary of an ML prediction result.

    Args
    ----
    prediction   The dict returned by engine_ml.predict() — must contain
                 cancer/metabolic/cardio/hematologic risk dicts + top_features.
    patient_info Dict with: age, sex, ethnicity (used for contextualisation).
    audience     "patient" (plain English, no jargon) or "clinician" (medical).

    Returns
    -------
    A 3–5 sentence narrative string, or None if Claude is unavailable.
    """
    cancer     = prediction.get("cancer", {})
    metabolic  = prediction.get("metabolic", {})
    cardio     = prediction.get("cardio", {})
    hematologic= prediction.get("hematologic", {})
    top_feats  = prediction.get("top_features", [])[:5]
    composite  = prediction.get("composite", 0)

    # Summarise features
    feat_lines = []
    for f in top_feats:
        direction = "↑ rising" if f.get("direction") == "risk_increasing" else "↓ declining"
        feat_lines.append(f"  - {f.get('label', f.get('feature', '?'))} ({direction})")
    feats_text = "\n".join(feat_lines) if feat_lines else "  (no specific drivers identified)"

    risk_summary = (
        f"Cancer risk: {cancer.get('level','?')} ({int(cancer.get('risk',0)*100)}%)\n"
        f"Metabolic risk: {metabolic.get('level','?')} ({int(metabolic.get('risk',0)*100)}%)\n"
        f"Cardiovascular risk: {cardio.get('level','?')} ({int(cardio.get('risk',0)*100)}%)\n"
        f"Hematologic risk: {hematologic.get('level','?')} ({int(hematologic.get('risk',0)*100)}%)\n"
        f"Composite score: {int(composite*100)}%\n"
        f"Top risk drivers:\n{feats_text}"
    )

    pat_context = (
        f"Patient: {patient_info.get('age','?')} year old "
        f"{patient_info.get('sex','?')}, {patient_info.get('ethnicity','?')}."
    )

    if audience == "patient":
        system = (
            "You are a compassionate health educator who explains medical risk assessments "
            "in language anyone can understand. You use simple words, avoid medical jargon, "
            "never cause alarm unnecessarily, and always encourage follow-up with a doctor. "
            "You write in short, clear sentences. You never claim to diagnose anything."
        )
        user = (
            f"{pat_context}\n\n"
            f"An AI health monitoring system has produced these risk scores:\n{risk_summary}\n\n"
            "Write 3–4 sentences for the patient explaining:\n"
            "1. What the overall picture looks like in simple terms.\n"
            "2. Which one or two readings are most worth discussing with their doctor.\n"
            "3. One practical, positive step they can take.\n"
            "Keep it warm, clear, and under 80 words. "
            "Do NOT reproduce the exact percentages — speak in plain English like "
            "'your blood sugar trend has been creeping up' not '53% metabolic risk'."
        )
    else:  # clinician
        system = (
            "You are an experienced clinical informatics assistant. "
            "You summarise AI-generated longitudinal health risk assessments for clinicians. "
            "Be concise, use appropriate medical terminology, reference the key biomarker drivers, "
            "and suggest concrete next clinical steps. Never diagnose — only flag for review."
        )
        user = (
            f"{pat_context}\n\n"
            f"Longitudinal risk assessment results:\n{risk_summary}\n\n"
            "Write a 4–5 sentence clinical summary covering:\n"
            "1. The dominant risk domains and their trajectory.\n"
            "2. The top 2–3 biomarker drivers flagged by the model.\n"
            "3. Suggested clinical actions (investigations, referrals, follow-up interval).\n"
            "Include a standard disclaimer that this is a decision-support tool, not a diagnosis."
        )

    return _call_claude(HAIKU_MODEL, [{"role": "user", "content": user}],
                        max_tokens=300, system=system)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  LONGITUDINAL TREND ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_trend_anomalies(
    patient_info: dict,
    checkups: list[dict],
    predictions_history: Optional[list[dict]] = None,
) -> Optional[str]:
    """
    Ask Claude Sonnet to analyse the full longitudinal record for unusual
    patterns that fixed numeric thresholds miss.

    Args
    ----
    patient_info         Dict: age, sex, ethnicity, family_history_*, smoking_status.
    checkups             List of checkup dicts ordered by date, each containing
                         biomarker fields. Dates should be strings "YYYY-MM-DD".
    predictions_history  Optional list of past prediction results for context.

    Returns
    -------
    A narrative string flagging anomalies, or None if Claude is unavailable.
    The output is intended for clinician review, not patient consumption.
    """
    if not checkups:
        return None

    # Summarise timeline compactly to stay within token limits
    timeline_lines = []
    key_markers = [
        "hba1c", "glucose_fasting", "hemoglobin", "lymphocytes_pct",
        "wbc", "cea", "alt", "ldl", "bp_systolic", "bmi", "crp", "psa",
        "ca125", "platelets", "creatinine", "tsh",
    ]
    for chk in checkups:
        date = chk.get("checkup_date", chk.get("date", "?"))
        vals = []
        for m in key_markers:
            v = chk.get(m)
            if v is not None:
                vals.append(f"{m}={v}")
        if vals:
            timeline_lines.append(f"  {date}: {', '.join(vals)}")

    timeline_text = "\n".join(timeline_lines)

    fam_hx = []
    if patient_info.get("family_history_cancer"):
        fam_hx.append("cancer")
    if patient_info.get("family_history_diabetes"):
        fam_hx.append("diabetes")
    if patient_info.get("family_history_cardio"):
        fam_hx.append("cardiovascular disease")
    fam_hx_str = ", ".join(fam_hx) if fam_hx else "none reported"

    ref_context = "\n".join(
        f"  {k}: normal range {v.get('normal','?')} {v.get('unit','')}"
        for k, v in BIOMARKER_REFS.items()
    )

    system = (
        "You are an expert clinical informatician specialising in longitudinal biomarker analysis. "
        "You detect subtle trends that no single test result reveals — the slow drift toward "
        "disease that clinicians miss between visits. You focus on TRAJECTORY and RATE OF CHANGE, "
        "not just current values. You flag patterns that fall within 'normal' ranges individually "
        "but form a concerning trend over time. You are conservative: you only flag patterns "
        "that have genuine clinical significance based on published evidence. You never diagnose."
    )

    user = (
        f"Patient profile:\n"
        f"  Age: {patient_info.get('age','?')}, Sex: {patient_info.get('sex','?')}, "
        f"  Ethnicity: {patient_info.get('ethnicity','?')}\n"
        f"  Smoking: {patient_info.get('smoking_status','?')}\n"
        f"  Family history: {fam_hx_str}\n\n"
        f"Longitudinal biomarker data (chronological):\n{timeline_text}\n\n"
        f"Reference ranges for context:\n{ref_context}\n\n"
        "Analyse this longitudinal record. Identify:\n"
        "1. Any biomarker showing a consistent directional trend over 2+ visits "
        "   (even if individual values remain 'normal').\n"
        "2. Any combination of trends that together suggest early disease trajectory "
        "   (e.g. CEA slowly rising + lymphocytes declining + hemoglobin drifting down).\n"
        "3. Any value whose rate of change is accelerating.\n"
        "4. Any pattern inconsistent with the patient's demographics or family history.\n\n"
        "Format your response as:\n"
        "FLAGGED PATTERNS:\n"
        "  1. [Pattern name]: [1–2 sentences describing the trend and supporting data points]\n"
        "     Clinical significance: [Why this matters]\n"
        "     Suggested action: [Concrete next step for clinician]\n\n"
        "If no clinically significant patterns are detected, say 'No significant trend anomalies "
        "detected in this longitudinal record.'\n\n"
        "DISCLAIMER: This is a decision-support analysis only. Clinical judgment is required."
    )

    return _call_claude(SONNET_MODEL, [{"role": "user", "content": user}],
                        max_tokens=700, system=system)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  DRUG INTERACTION PLAIN-ENGLISH EXPLANATION
# ══════════════════════════════════════════════════════════════════════════════

def explain_drug_interactions(
    drug_name: str,
    existing_medications: list[str],
    raw_fda_interactions: Optional[str] = None,
) -> Optional[str]:
    """
    Convert raw OpenFDA drug interaction text into a patient-friendly explanation.

    Args
    ----
    drug_name              The new drug being added.
    existing_medications   List of current medication names.
    raw_fda_interactions   Raw text from the OpenFDA drug_interactions field (optional).

    Returns
    -------
    A 2–4 sentence plain-English explanation of relevant interactions, or None.
    """
    if not existing_medications:
        return None

    existing_str = ", ".join(existing_medications)

    fda_context = ""
    if raw_fda_interactions:
        # Truncate to keep tokens manageable
        fda_context = f"\nOpenFDA drug interaction data:\n{raw_fda_interactions[:1500]}\n"

    system = (
        "You are a pharmacist assistant that explains drug interactions in plain English. "
        "You only flag interactions that are clinically meaningful. "
        "You never diagnose or instruct the patient to stop medication. "
        "You always recommend consulting a pharmacist or doctor. "
        "If interactions are minor or unlikely, say so clearly."
    )

    user = (
        f"A patient is being prescribed: {drug_name}\n"
        f"They are currently taking: {existing_str}\n"
        f"{fda_context}\n"
        "In 2–3 sentences: are there any important interactions to know about? "
        "Use simple, clear language. If there are no significant interactions, say so briefly. "
        "End with: 'Always confirm with your pharmacist or doctor before starting new medication.'"
    )

    return _call_claude(HAIKU_MODEL, [{"role": "user", "content": user}],
                        max_tokens=200, system=system)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  HEALTH CHECK
# ══════════════════════════════════════════════════════════════════════════════

def claude_ai_status() -> dict:
    """
    Return the status of Claude AI integration.
    Used by the /health and /api/v1/system-info endpoints.
    """
    key = _api_key()
    return {
        "available": bool(key),
        "key_configured": bool(key),
        "vision_model": SONNET_MODEL,
        "narrative_model": HAIKU_MODEL,
        "features": {
            "lab_report_ocr":           bool(key),
            "prediction_narrative":     bool(key),
            "trend_anomaly_detection":  bool(key),
            "drug_interaction_explain": bool(key),
        },
        "note": (
            "Set ANTHROPIC_API_KEY environment variable to enable AI features."
            if not key else
            "Claude AI fully operational."
        ),
    }
