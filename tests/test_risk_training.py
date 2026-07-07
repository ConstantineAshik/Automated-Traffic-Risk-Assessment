import pandas as pd

from risk_features import FEATURE_COLUMNS, frame_to_features
from training.evaluate_risk_model import evaluate_table
from training.train_risk_model import _validate, train


def test_frame_features_have_stable_schema():
    features = frame_to_features(
        {
            "ego_speed": "fast",
            "ttc_status": "closing_in",
            "objects": ["person", "truck"],
            "proximity_score": 0.5,
            "night": True,
        }
    )
    assert list(features) == FEATURE_COLUMNS
    assert features["heavy_vehicle_count"] == 1
    assert features["pedestrian_count"] == 1
    assert features["night"] == 1


def test_grouped_training_produces_three_class_predictions():
    rows = []
    for ride_index in range(8):
        for label_index, label in enumerate(("SAFE", "CAUTION", "DANGER")):
            row = frame_to_features(
                {
                    "ego_speed": ("stationary", "slow", "fast")[label_index],
                    "ttc_status": ("stable", "closing_in", "critical_approach")[
                        label_index
                    ],
                    "proximity_score": label_index / 2,
                    "objects": ["truck"] if label_index == 2 else [],
                    "is_erratic": label_index == 2,
                }
            )
            row.update(
                {
                    "ride_id": f"ride-{ride_index}",
                    "human_label": label,
                }
            )
            rows.append(row)

    table = _validate(pd.DataFrame(rows))
    model, metrics = train(table, test_size=0.25)
    assert set(model.classes_) == {0, 1, 2}
    assert metrics["train_rides"] == 6
    assert metrics["test_rides"] == 2


def test_validate_ignores_blank_labels_and_normalizes_values():
    rows = []
    for ride_index in range(3):
        for human_label in (" safe ", "1", "DANGER", ""):
            row = frame_to_features({"ego_speed": "slow", "objects": []})
            row.update(
                {
                    "ride_id": f"ride-{ride_index}",
                    "human_label": human_label,
                }
            )
            rows.append(row)

    table = _validate(pd.DataFrame(rows))
    assert len(table) == 9
    assert set(table["human_label"]) == {"SAFE", "CAUTION", "DANGER"}
    assert set(table["target"]) == {0, 1, 2}


def test_evaluate_table_reports_danger_recall_and_false_alert_rate():
    rows = []
    for index, (human_label, prediction) in enumerate(
        [
            ("SAFE", 0),
            ("CAUTION", 2),
            ("DANGER", 2),
            ("DANGER", 1),
        ]
    ):
        row = frame_to_features({"ego_speed": "fast", "objects": []})
        row.update(
            {
                "ride_id": "ride-1",
                "timestamp_seconds": index,
                "human_label": human_label,
                "smoothed_label": prediction,
            }
        )
        rows.append(row)

    metrics = evaluate_table(pd.DataFrame(rows))
    assert metrics["rows"] == 4
    assert metrics["danger_recall"] == 0.5
    assert metrics["danger_precision"] == 0.5
    assert metrics["false_danger_per_minute"] == 20.0
