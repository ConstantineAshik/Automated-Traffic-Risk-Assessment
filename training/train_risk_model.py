import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from risk_features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES


LABEL_MAPPING = {"SAFE": 0, "CAUTION": 1, "DANGER": 2}
REVERSE_LABEL_MAPPING = {value: key for key, value in LABEL_MAPPING.items()}


def _load_tables(paths):
    tables = []
    for path in paths:
        table = pd.read_csv(path)
        table["_source_file"] = str(path)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def _normalize_labels(labels):
    normalized = labels.map(
        lambda value: ""
        if pd.isna(value)
        else str(value).strip().upper()
    )
    numeric_aliases = {"0": "SAFE", "1": "CAUTION", "2": "DANGER"}
    return normalized.map(lambda value: numeric_aliases.get(value, value))


def _validate(table, min_ride_groups=3, require_all_classes=True):
    required = set(FEATURE_COLUMNS) | {"human_label", "ride_id"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    labels = _normalize_labels(table["human_label"])
    invalid = sorted(
        value for value in set(labels) if value and value not in LABEL_MAPPING
    )
    if invalid:
        raise ValueError(f"Invalid human_label values: {', '.join(invalid)}")
    labeled_mask = labels.isin(LABEL_MAPPING)
    table = table.loc[labeled_mask].copy()
    labels = labels.loc[labeled_mask]
    if table.empty:
        raise ValueError("No labeled rows found. Fill human_label before training.")
    if table["ride_id"].nunique() < min_ride_groups:
        raise ValueError(
            f"At least {min_ride_groups} independent ride_id group(s) are required"
        )
    if require_all_classes and labels.nunique() < 3:
        raise ValueError("SAFE, CAUTION, and DANGER examples are all required")
    table["human_label"] = labels
    table["target"] = labels.map(LABEL_MAPPING)
    return table


def _build_pipeline():
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )


def train(table, test_size=0.2):
    splitter = GroupShuffleSplit(
        n_splits=50,
        test_size=test_size,
        random_state=42,
    )
    train_indices = None
    test_indices = None
    required_classes = {0, 1, 2}
    for candidate_train, candidate_test in splitter.split(
        table, table["target"], groups=table["ride_id"]
    ):
        if (
            set(table.iloc[candidate_train]["target"]) == required_classes
            and set(table.iloc[candidate_test]["target"]) == required_classes
        ):
            train_indices, test_indices = candidate_train, candidate_test
            break
    if train_indices is None or test_indices is None:
        raise ValueError(
            "Could not create a ride-grouped holdout containing all three "
            "classes. Collect more class-diverse rides."
        )
    train_table = table.iloc[train_indices]
    test_table = table.iloc[test_indices]

    model = _build_pipeline()
    model.fit(train_table[FEATURE_COLUMNS], train_table["target"])
    predictions = model.predict(test_table[FEATURE_COLUMNS])
    metrics = {
        "accuracy": accuracy_score(test_table["target"], predictions),
        "balanced_accuracy": balanced_accuracy_score(
            test_table["target"], predictions
        ),
        "macro_f1": f1_score(
            test_table["target"], predictions, average="macro"
        ),
        "confusion_matrix": confusion_matrix(
            test_table["target"], predictions, labels=[0, 1, 2]
        ).tolist(),
        "classification_report": classification_report(
            test_table["target"],
            predictions,
            labels=[0, 1, 2],
            target_names=["SAFE", "CAUTION", "DANGER"],
            output_dict=True,
            zero_division=0,
        ),
        "train_rows": len(train_table),
        "test_rows": len(test_table),
        "train_rides": int(train_table["ride_id"].nunique()),
        "test_rides": int(test_table["ride_id"].nunique()),
    }
    return model, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train a grouped structured risk-classification baseline."
    )
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("models/risk_model.joblib"))
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    table = _validate(_load_tables(args.csv))
    model, metrics = train(table, test_size=args.test_size)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "label_mapping": LABEL_MAPPING,
        "metrics": metrics,
        "training_files": [str(path) for path in args.csv],
    }
    artifact = {
        "pipeline": model,
        "feature_columns": FEATURE_COLUMNS,
        "metadata": metadata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output)
    metrics_path = args.output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved model: {args.output}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
