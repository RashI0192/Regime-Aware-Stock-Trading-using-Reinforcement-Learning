"""Verify the saved model on the original chronological test segment."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from gym_anytrading.datasets import STOCKS_GOOGL
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from deployment_pipeline import ALL_FEATURES, build_features

ROOT = Path(__file__).resolve().parent
frame = STOCKS_GOOGL.copy().sort_index()
metadata = json.loads((ROOT / "models/logistic_top8_metadata.json").read_text())
assert hashlib.sha256(frame.to_csv().encode()).hexdigest() == metadata["dataset_sha256"]
features = build_features(frame)
assert features.columns.tolist() == ALL_FEATURES
assert features.iloc[252:-1].notna().all().all()
model = joblib.load(ROOT / "models/logistic_top8.joblib")
forward = frame.Open.shift(-1) / frame.Open - 1
labels = (forward > 0).astype(int)
test = np.arange(int(.8 * len(frame)), len(frame) - 1)
prob = model.predict_proba(features.iloc[test][ALL_FEATURES])[:, 1]
pred = (prob >= metadata["decision_threshold"]).astype(int)
print(json.dumps({
    "test_rows": len(test),
    "accuracy": accuracy_score(labels.iloc[test], pred),
    "balanced_accuracy": balanced_accuracy_score(labels.iloc[test], pred),
    "probability_min": float(prob.min()),
    "probability_max": float(prob.max()),
    "dataset_hash_verified": True,
    "feature_schema_verified": True,
}, indent=2))
