# Changelog

All notable changes to BioSentinel will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned for v0.2.0
- BioSentinel Transformer (BST) v1.0 — full temporal attention model
- Federated learning support (Phase 3 preview)
- DICOM imaging integration
- Mobile-responsive patient dashboard

---

## [0.1.0-alpha] — 2025-03-16

### 🎉 Initial Release

This is the first public release of BioSentinel. Foundations are laid; the mission begins.

#### Added
- **Core data model**: Patient, Checkup, LabResults, Medications, Diagnoses, DietPlan schemas
- **FHIR R4 mapping**: Full mapping to/from HL7 FHIR R4 resources
- **REST API (FastAPI)**: CRUD endpoints for all core entities
- **Authentication**: JWT-based auth with role-based access (patient, clinician, admin)
- **CancerRiskNet v0.1**: XGBoost-based pan-cancer risk ensemble (proof of concept)
- **Metabolic Module v0.1**: Type 2 diabetes and metabolic syndrome risk
- **Cardiovascular Module v0.1**: CVD risk scoring from longitudinal vitals + labs
- **Biomarker Normalizer**: Age/sex-adjusted reference range normalization for 120+ biomarkers
- **Missing Data Imputer**: KNN-based imputation for sparse longitudinal records
- **Basic alert system**: Risk level classification (Low/Moderate/High/Critical)
- **Patient Dashboard v0.1** (React): Timeline view, risk scores, lab trends
- **Docker Compose** deployment stack
- **GitHub Actions CI** pipeline with automated testing
- **Sample data**: Synthetic patient records for development/testing
- **Documentation**: Architecture overview, API reference, data schema docs

#### Known Limitations
- CancerRiskNet v0.1 uses proof-of-concept weights (not clinically validated)
- SHAP explainability integration is incomplete in this release
- Federated learning is not yet implemented
- Mobile app not yet available
- Imaging (DICOM) data not yet supported

---

## Contributing to the Changelog

When submitting a PR, please add your changes to the `[Unreleased]` section above following the format:

```markdown
### Added / Changed / Deprecated / Removed / Fixed / Security
- Brief description of change ([#123](link-to-pr))
```
