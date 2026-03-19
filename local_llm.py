"""
BioSentinel — Local LLM Integration via Ollama
================================================
Provides the exact same interface as claude_ai.py but uses a locally-running
Ollama instance instead of the Anthropic cloud API.

This makes BioSentinel 100% offline — no API keys, no cloud dependency,
no data ever leaving your server.

Supported models (recommended):
  llama3.2:3b     — fast, good for narratives on CPU (1.9GB)
  llama3.1:8b     — balanced quality/speed (4.7GB)
  llama3.1:70b    — best quality, requires GPU (40GB)
  mistral:7b      — great for clinical text (4.1GB)
  phi3:mini       — very fast, lightweight (2.3GB)
  gemma2:9b       — strong reasoning (5.5GB)

For vision OCR (lab report reading):
  llava:7b        — vision model, reads images (4.7GB)
  llava:13b       — better accuracy (8.0GB)
  minicpm-v:8b    — compact vision model (5.0GB)

Setup
-----
  # Install Ollama (Mac/Linux/Windows)
  curl -fsSL https://ollama.com/install.sh | sh

  # Pull models
  ollama pull llama3.1:8b        # for text tasks
  ollama pull llava:7b           # for vision/OCR tasks

  # Start Ollama (usually auto-starts)
  ollama serve

  # Configure BioSentinel (.env)
  LOCAL_LLM_ENABLED=1
  LOCAL_LLM_URL=http://localhost:11434
  LOCAL_LLM_TEXT_MODEL=llama3.1:8b
  LOCAL_LLM_VISION_MODEL=llava:7b

Environment variables
---------------------
LOCAL_LLM_ENABLED      — set to "1" to use Ollama instead of Claude API
LOCAL_LLM_URL          — Ollama server URL (default: http://localhost:11434)
LOCAL_LLM_TEXT_MODEL   — model for text tasks (default: llama3.1:8b)
LOCAL_LLM_VISION_MODEL — model for vision/OCR (default: llava:7b)
LOCAL_LLM_TIMEOUT      — request timeout in seconds (default: 120)
"""

import base64
import json
import logging
import os
from typing import Optional

logger = logging.getLogger("biosentinel.local_llm")

LOCAL_LLM_ENABLED = os.getenv("LOCAL_LLM_ENABLED", "0") == "1"
OLLAMA_URL        = os.getenv("LOCAL_LLM_URL", "http://localhost:11434")
TEXT_MODEL        = os.getenv("LOCAL_LLM_TEXT_MODEL", "llama3.1:8b")
VISION_MODEL      = os.getenv("LOCAL_LLM_VISION_MODEL", "llava:7b")
TIMEOUT           = int(os.getenv("LOCAL_LLM_TIMEOUT", "120"))

# ── Core Ollama client ────────────────────────────────────────────────────────

def _ollama_generate(
    model: str,
    prompt: str,
    system: Optional[str] = None,
    images: Optional[list] = None,
    stream: bool = False,
) -> Optional[str]:
    """Call the Ollama /api/generate endpoint."""
    try:
        import httpx
    except ImportError:
        logger.error("httpx not installed. Run: pip install httpx")
        return None

    payload: dict = {
        "model":  model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 800,
            "top_p": 0.9,
        },
    }
    if system:
        payload["system"] = system
    if images:
        payload["images"] = images  # base64 strings

    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            resp = client.post(f"{OLLAMA_URL}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "").strip()
    except Exception as e:
        logger.error(f"Ollama request failed: {e}")
        return None


def _ollama_chat(
    model: str,
    messages: list,
    system: Optional[str] = None,
) -> Optional[str]:
    """Call the Ollama /api/chat endpoint (supports multi-turn)."""
    try:
        import httpx
    except ImportError:
        return None

    chat_messages = []
    if system:
        chat_messages.append({"role": "system", "content": system})
    chat_messages.extend(messages)

    payload = {
        "model":    model,
        "messages": chat_messages,
        "stream":   False,
        "options":  {"temperature": 0.3, "top_p": 0.9},
    }

    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            resp = client.post(f"{OLLAMA_URL}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "").strip()
    except Exception as e:
        logger.error(f"Ollama chat failed: {e}")
        return None


# ── Status & model management ─────────────────────────────────────────────────

def local_llm_status() -> dict:
    """Return Ollama status and available models."""
    if not LOCAL_LLM_ENABLED:
        return {
            "enabled": False,
            "available": False,
            "note": "Set LOCAL_LLM_ENABLED=1 in .env to use local models.",
            "setup_url": "https://ollama.com/install",
        }

    try:
        import httpx
        with httpx.Client(timeout=5) as client:
            resp = client.get(f"{OLLAMA_URL}/api/tags")
            resp.raise_for_status()
            models_raw = resp.json().get("models", [])
            models = [m["name"] for m in models_raw]
    except Exception as e:
        return {
            "enabled":   True,
            "available": False,
            "ollama_url": OLLAMA_URL,
            "error": str(e),
            "note": f"Ollama not reachable at {OLLAMA_URL}. Is it running? Run: ollama serve",
        }

    text_ready   = any(TEXT_MODEL.split(":")[0] in m for m in models)
    vision_ready = any(VISION_MODEL.split(":")[0] in m for m in models)

    return {
        "enabled":       True,
        "available":     True,
        "ollama_url":    OLLAMA_URL,
        "text_model":    TEXT_MODEL,
        "vision_model":  VISION_MODEL,
        "models_pulled": models,
        "text_ready":    text_ready,
        "vision_ready":  vision_ready,
        "features": {
            "lab_report_ocr":          vision_ready,
            "prediction_narrative":    text_ready,
            "trend_anomaly_detection": text_ready,
            "drug_interaction_explain":text_ready,
        },
        "note": (
            "Local LLM fully operational." if (text_ready and vision_ready)
            else f"Pull missing models: ollama pull {TEXT_MODEL}"
              + (f" && ollama pull {VISION_MODEL}" if not vision_ready else "")
        ),
    }


def pull_model(model_name: str) -> dict:
    """
    Pull (download) a model via Ollama.
    Returns immediately with a status — actual download happens in background.
    Monitor with: ollama list
    """
    try:
        import httpx
        with httpx.Client(timeout=10) as client:
            resp = client.post(
                f"{OLLAMA_URL}/api/pull",
                json={"name": model_name, "stream": False},
            )
            if resp.status_code == 200:
                return {"status": "pulling", "model": model_name,
                        "message": f"Pulling {model_name}. Check progress with: ollama list"}
            return {"status": "error", "model": model_name, "detail": resp.text[:200]}
    except Exception as e:
        return {"status": "error", "model": model_name, "detail": str(e)}


# ── Lab report OCR ────────────────────────────────────────────────────────────

def extract_labs_from_image_local(
    image_bytes: bytes,
    media_type: str = "image/jpeg",
) -> dict:
    """
    Extract biomarker values from a lab report image using a local vision model.
    Uses llava or similar Ollama vision model.
    """
    if not LOCAL_LLM_ENABLED:
        return {"error": "Local LLM not enabled. Set LOCAL_LLM_ENABLED=1.", "values": {}}

    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    prompt = (
        "You are a medical data extraction system. "
        "Look at this lab report image and extract all numeric biomarker values. "
        "Return ONLY a JSON object with field names and numeric values. "
        "Use these exact field names where possible: "
        "hba1c, glucose_fasting, hemoglobin, wbc, platelets, lymphocytes_pct, "
        "neutrophils_pct, cea, ca125, psa, alt, ast, ldl, hdl, total_cholesterol, "
        "triglycerides, creatinine, tsh, crp, ferritin, vitamin_d, vitamin_b12, "
        "bp_systolic, bp_diastolic, bmi, weight_kg, albumin, bilirubin_total. "
        "Only include fields that are actually present in the image. "
        "Return ONLY the JSON, no other text."
    )

    raw = _ollama_generate(VISION_MODEL, prompt, images=[b64])
    if raw is None:
        return {"error": f"Vision model {VISION_MODEL} failed", "values": {}}

    # Strip markdown fences
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw

    try:
        parsed = json.loads(raw)
        values = {k: float(v) for k, v in parsed.items()
                  if isinstance(v, (int, float)) and v is not None}
        return {"values": values, "method": "local_llm_vision", "model": VISION_MODEL}
    except json.JSONDecodeError as e:
        logger.error(f"Vision JSON parse error: {e}\nRaw: {raw[:300]}")
        return {"error": f"JSON parse failed: {e}", "values": {}, "raw": raw[:300]}


# ── Prediction narrative ──────────────────────────────────────────────────────

def generate_prediction_narrative_local(
    prediction: dict,
    patient_info: dict,
    audience: str = "patient",
) -> Optional[str]:
    """Generate plain-English prediction narrative using local LLM."""
    if not LOCAL_LLM_ENABLED:
        return None

    cancer     = prediction.get("cancer", {})
    metabolic  = prediction.get("metabolic", {})
    cardio     = prediction.get("cardio", {})
    hematologic= prediction.get("hematologic", {})
    top_feats  = prediction.get("top_features", [])[:5]

    feat_lines = "\n".join(
        f"  - {f.get('label', f.get('feature', '?'))} "
        f"({'rising' if f.get('direction')=='risk_increasing' else 'declining'})"
        for f in top_feats
    ) or "  (no specific drivers)"

    risk_text = (
        f"Cancer: {cancer.get('level','?')} ({int(cancer.get('risk',0)*100)}%)\n"
        f"Metabolic: {metabolic.get('level','?')} ({int(metabolic.get('risk',0)*100)}%)\n"
        f"Cardiovascular: {cardio.get('level','?')} ({int(cardio.get('risk',0)*100)}%)\n"
        f"Hematologic: {hematologic.get('level','?')} ({int(hematologic.get('risk',0)*100)}%)\n"
        f"Key factors:\n{feat_lines}"
    )

    pat = (
        f"Patient: {patient_info.get('age','?')} year old "
        f"{patient_info.get('sex','?')}, {patient_info.get('ethnicity','?')}."
    )

    if audience == "patient":
        system = (
            "You are a compassionate health educator. Explain medical risk results "
            "in simple, jargon-free language. Never cause alarm unnecessarily. "
            "Always recommend consulting a doctor. Maximum 80 words."
        )
        prompt = (
            f"{pat}\n\nAI health assessment results:\n{risk_text}\n\n"
            "Write 3-4 sentences for the patient: what the overall picture looks like, "
            "one reading worth discussing with their doctor, and one practical positive step. "
            "Use plain language, no medical jargon, no exact percentages."
        )
    else:
        system = (
            "You are a clinical informatics assistant summarising AI risk assessments. "
            "Be concise, use medical terminology, reference biomarker drivers, suggest actions. "
            "Include disclaimer that this is decision-support, not diagnosis."
        )
        prompt = (
            f"{pat}\n\nLongitudinal risk assessment:\n{risk_text}\n\n"
            "Write 4-5 sentence clinical summary: dominant risk domains, "
            "top biomarker drivers, suggested clinical actions. "
            "End with standard decision-support disclaimer."
        )

    return _ollama_chat(TEXT_MODEL, [{"role": "user", "content": prompt}], system=system)


# ── Trend anomaly detection ───────────────────────────────────────────────────

def detect_trend_anomalies_local(
    patient_info: dict,
    checkups: list,
    predictions_history: Optional[list] = None,
) -> Optional[str]:
    """Detect longitudinal biomarker anomalies using local LLM."""
    if not LOCAL_LLM_ENABLED or not checkups:
        return None

    key_markers = [
        "hba1c", "glucose_fasting", "hemoglobin", "lymphocytes_pct",
        "wbc", "cea", "alt", "ldl", "bp_systolic", "bmi", "crp",
    ]
    timeline = []
    for chk in checkups:
        date = chk.get("checkup_date", chk.get("date", "?"))
        vals = ", ".join(
            f"{m}={chk[m]}"
            for m in key_markers
            if chk.get(m) is not None
        )
        if vals:
            timeline.append(f"  {date}: {vals}")

    fam_hx = []
    for k in ("family_history_cancer", "family_history_diabetes", "family_history_cardio"):
        if patient_info.get(k):
            fam_hx.append(k.replace("family_history_", "").replace("_", " "))

    system = (
        "You are a clinical informatician specialising in longitudinal biomarker analysis. "
        "Find subtle multi-visit trends that single tests miss. "
        "Focus on rate of change, not just absolute values. "
        "Flag patterns with clinical significance. "
        "Be concise. Never diagnose — only flag for review."
    )

    prompt = (
        f"Patient: {patient_info.get('age','?')}yr {patient_info.get('sex','?')}, "
        f"ethnicity: {patient_info.get('ethnicity','?')}, "
        f"family history: {', '.join(fam_hx) or 'none'}\n\n"
        f"Biomarker timeline:\n" + "\n".join(timeline) + "\n\n"
        "Analyse for: 1) Consistent directional trends across 2+ visits "
        "(even if values remain 'normal'). 2) Multi-marker combinations suggesting "
        "early disease trajectory. 3) Accelerating rates of change.\n\n"
        "Format:\nFLAGGED PATTERNS:\n  1. [Name]: [trend description]\n"
        "     Clinical significance: [why it matters]\n"
        "     Suggested action: [next step]\n\n"
        "If no patterns: say 'No significant trend anomalies detected.'\n"
        "DISCLAIMER: Decision-support only. Clinical judgment required."
    )

    return _ollama_chat(TEXT_MODEL, [{"role": "user", "content": prompt}], system=system)


# ── Drug interaction explanation ──────────────────────────────────────────────

def explain_drug_interactions_local(
    drug_name: str,
    existing_medications: list,
    raw_fda_interactions: Optional[str] = None,
) -> Optional[str]:
    """Explain drug interactions using local LLM."""
    if not LOCAL_LLM_ENABLED or not existing_medications:
        return None

    fda_ctx = (
        f"\nOpenFDA interaction data:\n{raw_fda_interactions[:800]}"
        if raw_fda_interactions else ""
    )

    system = (
        "You are a pharmacist assistant. Explain drug interactions in plain English. "
        "Only flag clinically meaningful interactions. Never instruct stopping medication. "
        "Always recommend consulting pharmacist or doctor."
    )

    prompt = (
        f"New drug: {drug_name}\n"
        f"Current medications: {', '.join(existing_medications)}\n"
        f"{fda_ctx}\n\n"
        "In 2-3 sentences: any important interactions? Use simple language. "
        "If no significant interactions, say so briefly. "
        "End with: 'Always confirm with your pharmacist or doctor.'"
    )

    return _ollama_chat(TEXT_MODEL, [{"role": "user", "content": prompt}], system=system)


# ── Unified AI interface (auto-selects local vs cloud) ────────────────────────

def get_ai_backend() -> str:
    """Return 'local' if Ollama is enabled and reachable, else 'claude'."""
    if not LOCAL_LLM_ENABLED:
        return "claude"
    status = local_llm_status()
    if status.get("available"):
        return "local"
    return "claude"
