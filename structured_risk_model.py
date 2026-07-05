from pathlib import Path
from typing import Dict, Iterable, List

import joblib
import pandas as pd

from risk_features import FEATURE_COLUMNS, frames_to_features


class StructuredRiskModel:
    """Loads a risk classifier trained by training/train_risk_model.py."""

    def __init__(self, model_path: str):
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Risk model not found: {path}")
        artifact = joblib.load(path)
        if not isinstance(artifact, dict) or "pipeline" not in artifact:
            raise ValueError("Invalid structured risk-model artifact")
        self.pipeline = artifact["pipeline"]
        self.feature_columns = artifact.get("feature_columns", FEATURE_COLUMNS)
        self.metadata: Dict = artifact.get("metadata", {})

    def predict(self, frame_data: Iterable[Dict]) -> List[int]:
        rows = frames_to_features(frame_data)
        table = pd.DataFrame(rows, columns=self.feature_columns)
        return [int(value) for value in self.pipeline.predict(table)]

    def predict_proba(self, frame_data: Iterable[Dict]):
        rows = frames_to_features(frame_data)
        table = pd.DataFrame(rows, columns=self.feature_columns)
        return self.pipeline.predict_proba(table)
