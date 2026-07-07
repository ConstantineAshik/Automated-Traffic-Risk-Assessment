import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from risk_features import FEATURE_COLUMNS
from training.train_risk_model import (
    LABEL_MAPPING,
    REVERSE_LABEL_MAPPING,
    _load_tables,
    _validate,
)


def _coerce_predictions(values):
    labels = []
    for value in values:
        if pd.isna(value):
            raise ValueError("Prediction column contains blank values")
        text = str(value).strip().upper()
        if text in LABEL_MAPPING:
            labels.append(LABEL_MAPPING[text])
            continue
        try:
            numeric = int(float(text))
        except ValueError as exc:
            raise ValueError(f"Invalid prediction value: {value}") from exc
        if numeric not in REVERSE_LABEL_MAPPING:
            raise ValueError(f"Invalid prediction value: {value}")
        labels.append(numeric)
    return labels


def _false_danger_per_minute(table, predictions):
    false_danger = sum(
        1
        for actual, predicted in zip(table["target"], predictions)
        if int(predicted) == 2 and int(actual) != 2
    )
    if "timestamp_seconds" not in table.columns:
        return None

    duration_seconds = 0.0
    for _, ride in table.groupby("ride_id"):
        timestamps = pd.to_numeric(ride["timestamp_seconds"], errors="coerce").dropna()
        if timestamps.empty:
            continue
        observed = max(float(timestamps.max() - timestamps.min()), 0.0)
        # CSVs are emitted at the configured sampling rate, usually 2 FPS. If a
        # tiny test clip has only one or two timestamps, this lower bound keeps
        # the rate finite and conservative.
        lower_bound = len(ride) / 2.0
        duration_seconds += max(observed, lower_bound)

    if duration_seconds <= 0:
        return None
    return false_danger / (duration_seconds / 60.0)


def _predict_with_model(table, model_path):
    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict) or "pipeline" not in artifact:
        raise ValueError(f"Invalid structured risk-model artifact: {model_path}")
    feature_columns = artifact.get("feature_columns", FEATURE_COLUMNS)
    return [int(value) for value in artifact["pipeline"].predict(table[feature_columns])]


def evaluate_table(table, prediction_column="smoothed_label", model_path=None):
    table = _validate(
        table,
        min_ride_groups=1,
        require_all_classes=False,
    )
    if model_path is not None:
        predictions = _predict_with_model(table, model_path)
        evaluated_source = str(model_path)
    else:
        if prediction_column not in table.columns:
            raise ValueError(f"Missing prediction column: {prediction_column}")
        predictions = _coerce_predictions(table[prediction_column])
        evaluated_source = prediction_column

    actual = [int(value) for value in table["target"]]
    report = classification_report(
        actual,
        predictions,
        labels=[0, 1, 2],
        target_names=["SAFE", "CAUTION", "DANGER"],
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "evaluated_source": evaluated_source,
        "rows": int(len(table)),
        "rides": int(table["ride_id"].nunique()),
        "accuracy": accuracy_score(actual, predictions),
        "balanced_accuracy": balanced_accuracy_score(actual, predictions),
        "macro_f1": f1_score(actual, predictions, labels=[0, 1, 2], average="macro"),
        "danger_precision": report["DANGER"]["precision"],
        "danger_recall": report["DANGER"]["recall"],
        "false_danger_per_minute": _false_danger_per_minute(table, predictions),
        "confusion_matrix": confusion_matrix(
            actual,
            predictions,
            labels=[0, 1, 2],
        ).tolist(),
        "label_order": ["SAFE", "CAUTION", "DANGER"],
        "classification_report": report,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate exported prediction CSVs against filled human_label values."
        )
    )
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument(
        "--prediction-column",
        default="smoothed_label",
        help="CSV prediction column to evaluate when --model is not supplied.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="Optional models/risk_model.joblib artifact to evaluate directly.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON metrics path.")
    args = parser.parse_args()

    metrics = evaluate_table(
        _load_tables(args.csv),
        prediction_column=args.prediction_column,
        model_path=args.model,
    )
    metrics_json = json.dumps(metrics, indent=2)
    print(metrics_json)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(metrics_json + "\n", encoding="utf-8")
        print(f"Saved metrics: {args.output}")


if __name__ == "__main__":
    main()
