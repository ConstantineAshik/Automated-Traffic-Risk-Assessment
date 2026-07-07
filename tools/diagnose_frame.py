r"""
Simple diagnostic helper to evaluate a single synthetic `frame_data` dict:
- prints the description emitted by `TextGenerator`
- prints numeric score from `RiskCalculator`
- optionally prints prediction from a trained structured model

Usage:
  python tools\diagnose_frame.py --model models\risk_model.joblib

Edit the `sample_frame` dict below to match the suspected frame values.
"""
import json
import os
import sys

# ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from risk_features import frame_to_features
from risk_calculator import RiskCalculator
from structured_risk_model import StructuredRiskModel
from text_generator import TextGenerator


LABELS = {0: "SAFE", 1: "CAUTION", 2: "DANGER"}


def diagnose(frame_data, model_path=None):
    tg = TextGenerator()
    rc = RiskCalculator()

    desc = tg.generate_description(frame_data)
    speed_cat = frame_data.get("ego_speed", "slow")
    score = rc.calculate_risk_score(frame_data, speed_cat)

    print("DESCRIPTION:")
    print(desc)
    print()
    print("FEATURES:")
    print(json.dumps(frame_to_features(frame_data), indent=2))
    print()
    print("NUMERIC SCORE:", score)
    print("NUMERIC LABEL:", rc.score_to_label(score))
    if model_path:
        model = StructuredRiskModel(model_path)
        prediction = model.predict([frame_data])[0]
        print("STRUCTURED MODEL:", prediction, "->", LABELS[prediction])
        try:
            probabilities = model.predict_proba([frame_data])[0]
            print(
                "MODEL PROBABILITIES:",
                {
                    LABELS[index]: round(float(value), 4)
                    for index, value in enumerate(probabilities)
                },
            )
        except AttributeError:
            pass


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Optional trained models/risk_model.joblib")
    args = parser.parse_args()

    # Example 1: sample frame with elevated risks
    sample_frame_1 = {
        "ego_speed": "fast",
        "is_erratic": False,
        "ttc_status": "critical_approach",
        "proximity_score": 0.4,
        "objects": ["person"],
        "pedestrian_crossing_risk": True,
        "wrong_side_risk": True,
        "phone_risk": "safe",
    }

    print("\n--- Diagnosis: SAMPLE FRAME 1 ---\n")
    diagnose(sample_frame_1, args.model)

    # Example 2: sample frame with lower risk signals
    sample_frame_2 = {
        "ego_speed": "fast",
        "is_erratic": False,
        "ttc_status": "closing_in",
        "proximity_score": 0.25,
        "objects": ["person"],
        "pedestrian_crossing_risk": True,
        "wrong_side_risk": False,
        "phone_risk": "safe",
    }

    print("\n--- Diagnosis: SAMPLE FRAME 2 ---\n")
    diagnose(sample_frame_2, args.model)
