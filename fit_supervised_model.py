"""Fit and save the selected logistic_top8 deployment artifact."""

import hashlib
import json
from pathlib import Path

import joblib
from gym_anytrading.datasets import STOCKS_GOOGL

from deployment_pipeline import ALL_FEATURES, build_features, make_logistic_top8


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "models"
OUT.mkdir(exist_ok=True)
frame = STOCKS_GOOGL.copy().sort_index()
features = build_features(frame)
forward = frame["Open"].shift(-1) / frame["Open"] - 1
labels = (forward > 0).astype(int)

# Match the original study: calibration/history [0,252), training [252,1401).
train_start, train_end = 252, int(0.6 * len(frame))
idx = range(train_start, train_end - 1)
X_train = features.iloc[list(idx)][ALL_FEATURES]
y_train = labels.iloc[list(idx)]
if X_train.isna().any().any():
    raise ValueError("Training features contain missing values")

model = make_logistic_top8().set_params(model__C=0.1)
model.fit(X_train, y_train)
joblib.dump(model, OUT / "logistic_top8.joblib")

metadata = {
    "model_version": "logistic_top8-research-train-v1",
    "model_name": "logistic_top8",
    "decision_threshold": 0.53,
    "training_rows": [train_start, train_end],
    "training_examples": len(y_train),
    "feature_columns": ALL_FEATURES,
    "selected_features": model.named_steps["select"].get_support().nonzero()[0].tolist(),
    "logistic_C": 0.1,
    "dataset_sha256": hashlib.sha256(frame.to_csv().encode()).hexdigest(),
    "source": "gym_anytrading.datasets.STOCKS_GOOGL",
    "note": "Research artifact; not evidence of prospective trading performance.",
}
(OUT / "logistic_top8_metadata.json").write_text(json.dumps(metadata, indent=2))
print(json.dumps(metadata, indent=2))
