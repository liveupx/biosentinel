"""
BioSentinel Base Disease Model
Abstract base class that all disease prediction modules must implement.

All models must:
- Accept a PatientTimeline as input
- Return a RiskPrediction with calibrated probability (0.0–1.0)
- Include SHAP values for explainability
- Implement validate_input() before prediction
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class RiskPrediction:
    """Standardized risk prediction output for all disease modules."""
    model_id: str
    disease: str
    disease_category: str
    risk_score: float          # Calibrated probability 0.0-1.0
    risk_level: str            # "low" | "moderate" | "high" | "critical"
    confidence: float          # Model confidence 0.0-1.0
    shap_values: dict          # Feature name → SHAP value
    top_features: list         # Top 5 feature explanations
    data_completeness: float   # % of expected features present
    lookback_months: int       # How many months of data were used
    checkups_used: int         # Number of checkups used
    recommendation: Optional[str] = None  # Clinical recommendation string
    alert_required: bool = False

    @property
    def risk_level_from_score(self) -> str:
        if self.risk_score < 0.25:
            return "low"
        elif self.risk_score < 0.50:
            return "moderate"
        elif self.risk_score < 0.75:
            return "high"
        else:
            return "critical"

    def __post_init__(self):
        self.risk_level = self.risk_level_from_score
        self.alert_required = self.risk_score >= 0.50


class InsufficientDataError(Exception):
    """Raised when patient timeline has insufficient data for prediction."""
    pass


class BaseDiseaseModel(ABC):
    """
    Abstract base class for all BioSentinel disease prediction modules.

    All subclasses must implement:
    - predict(timeline) -> RiskPrediction
    - _extract_features(timeline) -> dict
    - _run_model(features) -> float

    Subclasses should declare:
    - MODEL_ID: unique string identifier
    - DISEASE_NAME: human-readable disease name
    - DISEASE_CATEGORY: category (cancer, cardiovascular, metabolic, etc.)
    - TARGET_BIOMARKERS: list of key biomarker features
    - MIN_CHECKUPS_REQUIRED: minimum checkups needed for prediction
    """

    MODEL_ID: str = ""
    DISEASE_NAME: str = ""
    DISEASE_CATEGORY: str = ""
    TARGET_BIOMARKERS: list = field(default_factory=list)
    MIN_CHECKUPS_REQUIRED: int = 2
    MODEL_VERSION: str = "0.1.0"

    def __init__(self):
        self._model = None
        self._is_loaded = False
        logger.info(f"Initialized {self.__class__.__name__} (ID: {self.MODEL_ID})")

    def load(self) -> None:
        """Load model weights from disk. Called lazily on first prediction."""
        self._load_model_weights()
        self._is_loaded = True
        logger.info(f"Model {self.MODEL_ID} loaded successfully.")

    def _load_model_weights(self) -> None:
        """Override to implement model weight loading."""
        pass

    def _validate_input(self, timeline) -> None:
        """Validate that timeline has sufficient data for prediction."""
        if not timeline or not timeline.checkups:
            raise InsufficientDataError(
                f"Patient timeline is empty. "
                f"{self.DISEASE_NAME} model requires at least "
                f"{self.MIN_CHECKUPS_REQUIRED} checkups."
            )
        if len(timeline.checkups) < self.MIN_CHECKUPS_REQUIRED:
            raise InsufficientDataError(
                f"Insufficient checkups: {len(timeline.checkups)} found, "
                f"{self.MIN_CHECKUPS_REQUIRED} required for {self.DISEASE_NAME} prediction."
            )

    def _get_data_completeness(self, features: dict) -> float:
        """Calculate what fraction of expected features are present and non-null."""
        if not self.TARGET_BIOMARKERS:
            return 1.0
        present = sum(1 for b in self.TARGET_BIOMARKERS if features.get(b) is not None)
        return round(present / len(self.TARGET_BIOMARKERS), 2)

    @abstractmethod
    def predict(self, timeline) -> RiskPrediction:
        """
        Run disease risk prediction on a patient timeline.

        Args:
            timeline: PatientTimeline object with longitudinal health data.

        Returns:
            RiskPrediction with risk score, confidence, and SHAP explanation.

        Raises:
            InsufficientDataError: If timeline has insufficient checkups.
        """
        ...

    @abstractmethod
    def _extract_features(self, timeline) -> dict:
        """Extract temporal features from patient timeline."""
        ...

    @abstractmethod
    def _run_model(self, features: dict) -> float:
        """Run the ML model and return raw risk probability (0.0–1.0)."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.MODEL_ID}, loaded={self._is_loaded})"
