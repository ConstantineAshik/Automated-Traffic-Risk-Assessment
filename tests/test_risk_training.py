import pandas as pd

from risk_features import FEATURE_COLUMNS, frame_to_features
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
