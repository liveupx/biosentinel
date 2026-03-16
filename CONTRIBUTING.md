# Contributing to BioSentinel

First off, **thank you** for considering a contribution to BioSentinel. This is a project with a mission to save lives — every improvement, no matter how small, matters.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Code Style Guidelines](#code-style-guidelines)
- [Adding a New Disease Module](#adding-a-new-disease-module)
- [Pull Request Process](#pull-request-process)
- [Commit Message Convention](#commit-message-convention)
- [Medical & Ethical Guidelines](#medical--ethical-guidelines)

---

## Code of Conduct

This project and everyone participating in it is governed by the [BioSentinel Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold it. Report unacceptable behavior to [biosentinel@liveupx.com](mailto:biosentinel@liveupx.com).

---

## How Can I Contribute?

### 🐛 Reporting Bugs

- Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md)
- Include your OS, Python version, and reproduction steps
- For data-related bugs, provide a minimal anonymized sample

### 💡 Suggesting Enhancements

- Use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.md)
- Describe the clinical or research motivation clearly
- Reference any relevant literature if applicable

### 🔬 Contributing ML Models

New disease prediction modules are always welcome! See [Adding a New Disease Module](#adding-a-new-disease-module).

### 📖 Improving Documentation

Documentation PRs are extremely valuable. Even fixing typos is welcome.

### 🩺 Clinical / Medical Expertise

If you are a healthcare professional, your domain expertise is incredibly valuable for:
- Validating feature engineering logic (are we using biomarkers correctly?)
- Reviewing alert thresholds
- Suggesting clinically relevant prediction targets

---

## Development Setup

```bash
# 1. Fork the repo on GitHub, then clone your fork
git clone https://github.com/YOUR-USERNAME/biosentinel.git
cd biosentinel

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install all dependencies including dev tools
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 4. Install pre-commit hooks
pip install pre-commit
pre-commit install

# 5. Set up the database
createdb biosentinel_dev
python -m biosentinel.db migrate

# 6. Copy and configure environment
cp .env.example .env.dev
# Edit .env.dev

# 7. Run the test suite to verify setup
pytest tests/ -v

# 8. Start the development server
uvicorn src.api.main:app --reload
```

---

## Code Style Guidelines

We use the following tools, all enforced by pre-commit hooks:

| Tool | Purpose | Config |
|---|---|---|
| **Black** | Code formatting | `pyproject.toml` |
| **isort** | Import sorting | `pyproject.toml` |
| **mypy** | Static type checking | `mypy.ini` |
| **flake8** | Linting | `.flake8` |
| **pytest** | Testing | `pytest.ini` |

### Key Guidelines

1. **Type annotations are required** for all public functions
2. **Docstrings are required** for all classes and public methods (Google style)
3. **Test coverage**: New features must include tests. Minimum 80% coverage for new modules
4. **No hardcoded patient data** — ever. Use fixtures and synthetic data only
5. All ML models must include a `predict_proba()` method returning calibrated probabilities

```python
# Example: correct function signature
def predict_risk(
    patient_timeline: PatientTimeline,
    lookback_months: int = 24,
    confidence_threshold: float = 0.7
) -> DiseaseRiskResult:
    """
    Predict disease risk from longitudinal patient timeline.

    Args:
        patient_timeline: Patient health timeline object.
        lookback_months: Number of months of history to use.
        confidence_threshold: Minimum confidence to include a prediction.

    Returns:
        DiseaseRiskResult with risk scores, confidence, and SHAP values.

    Raises:
        InsufficientDataError: If timeline has fewer than 2 checkpoints.
    """
    ...
```

---

## Adding a New Disease Module

New disease prediction modules should follow this structure:

```
src/models/{disease_category}/{disease_name}_model.py
tests/models/test_{disease_name}_model.py
docs/models/{disease_name}.md
```

### Module Template

```python
# src/models/cancer/example_cancer_model.py

from biosentinel.models.base_model import BaseDiseaseModel
from biosentinel.schemas import PatientTimeline, RiskPrediction


class ExampleCancerModel(BaseDiseaseModel):
    """
    Prediction model for Example Cancer.

    Model Type: XGBoost + LSTM Ensemble
    Target Variable: Binary cancer risk within 24 months
    Training Data: [describe dataset used, or 'pending training data']
    Literature AUC: 0.XX (cite source)
    """

    MODEL_ID = "example_cancer_v1"
    DISEASE_NAME = "example_cancer"
    DISEASE_CATEGORY = "cancer"
    TARGET_BIOMARKERS = ["biomarker_1", "biomarker_2"]  # key features
    MIN_CHECKUPS_REQUIRED = 2

    def __init__(self):
        super().__init__()
        # Initialize model components

    def predict(self, timeline: PatientTimeline) -> RiskPrediction:
        """Run prediction on patient timeline."""
        self._validate_input(timeline)
        features = self._extract_features(timeline)
        score = self._run_ensemble(features)
        explanation = self._compute_shap(features)

        return RiskPrediction(
            model_id=self.MODEL_ID,
            disease=self.DISEASE_NAME,
            risk_score=score,
            confidence=self._get_confidence(timeline),
            shap_values=explanation,
            top_features=explanation.top_n(5)
        )

    def _extract_features(self, timeline: PatientTimeline) -> dict:
        """Extract temporal features from patient timeline."""
        ...

    def _run_ensemble(self, features: dict) -> float:
        """Run the ensemble model and return calibrated risk score 0-1."""
        ...
```

### Checklist for New Modules

- [ ] Follows `BaseDiseaseModel` interface
- [ ] `predict()` returns a `RiskPrediction` object with SHAP values
- [ ] Includes unit tests in `tests/models/`
- [ ] Includes literature citation in docstring
- [ ] Registered in `src/models/__init__.py`
- [ ] Model docs in `docs/models/{name}.md`
- [ ] **Does NOT** include any real patient data

---

## Pull Request Process

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feat/your-feature-name
   # or
   git checkout -b fix/your-bug-name
   ```

2. **Make your changes** with clear, focused commits

3. **Run the full test suite**:
   ```bash
   pytest tests/ -v --cov=src --cov-report=html
   ```

4. **Run pre-commit checks**:
   ```bash
   pre-commit run --all-files
   ```

5. **Push and open a PR** against `main` using our PR template

6. **Address review feedback** — all PRs require at least 1 approval

7. **Squash merge** will be used for clean history

---

## Commit Message Convention

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): description

[optional body]

[optional footer(s)]
```

**Types:**
- `feat` — New feature
- `fix` — Bug fix
- `docs` — Documentation only
- `model` — New or improved ML model
- `refactor` — Code refactoring
- `test` — Adding tests
- `chore` — Build/config changes
- `security` — Security fixes

**Examples:**
```
feat(api): add bulk patient ingestion endpoint
fix(preprocessing): handle missing HbA1c values in normalization
model(cancer): add colorectal cancer prediction module v1
docs: update FHIR integration guide
security: patch SQL injection in patient filter endpoint
```

---

## Medical & Ethical Guidelines

Because this project directly relates to human health, we hold contributors to a higher standard:

1. **Never fabricate clinical validity claims.** If a model hasn't been validated, say so clearly.
2. **Always cite your sources.** Clinical claims must reference peer-reviewed literature.
3. **Privacy first.** Never commit real patient data, even anonymized. Use synthetic data or public research datasets only (MIMIC-III/IV with proper authorization, etc.).
4. **Appropriate uncertainty.** All predictions must include confidence intervals. Overconfident models are dangerous.
5. **Bias awareness.** Consider demographic biases in training data. Document known limitations in model docs.
6. **No diagnostic language.** The platform is a decision-support tool, not a diagnostic tool. Documentation must reflect this.

---

## Questions?

Open a [Discussion](https://github.com/liveupx/biosentinel/discussions) or email [biosentinel@liveupx.com](mailto:biosentinel@liveupx.com).

**Thank you for helping build something that could genuinely save lives. ❤️**
