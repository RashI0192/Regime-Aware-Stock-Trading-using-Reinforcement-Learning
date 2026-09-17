"""Local prediction entry point for the fitted logistic_top8 model."""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from deployment_pipeline import predict_local


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "logistic_top8.joblib"
METADATA_PATH = ROOT / "models" / "logistic_top8_metadata.json"


def load_model():
    """Load the fitted model and its deployment metadata."""
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text())
    return model, metadata


def predict_frame(frame: pd.DataFrame, symbol: str = "GOOGL") -> dict:
    """Return a JSON-compatible prediction for the latest row in ``frame``."""
    model, metadata = load_model()
    result = predict_local(frame, model, metadata["decision_threshold"])
    result.update({
        "symbol": symbol,
        "model_version": metadata["model_version"],
        "as_of": frame.index[-1].isoformat() if hasattr(frame.index[-1], "isoformat") else str(frame.index[-1]),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    })
    return result


if __name__ == "__main__":
    from gym_anytrading.datasets import STOCKS_GOOGL
    print(predict_frame(STOCKS_GOOGL.copy()))
