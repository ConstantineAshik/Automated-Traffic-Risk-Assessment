"""
Simple diagnostic helper to evaluate a single synthetic `frame_data` dict:
- prints the description emitted by `TextGenerator`
- prints numeric score from `RiskCalculator`
- prints model prediction from `RiskModel` (trained mock)

Usage:
  python tools\diagnose_frame.py

Edit the `sample_frame` dict below to match the suspected frame values.
"""
import sys
import os
import json
# ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from text_generator import TextGenerator
from risk_calculator import RiskCalculator
from risk_model import RiskModel


def diagnose(frame_data):
    tg = TextGenerator()
    rc = RiskCalculator()
    rm = RiskModel()
    rm.train_mock_model()

    desc = tg.generate_description(frame_data)
    speed_cat = frame_data.get("ego_speed", "slow")
    if speed_cat == "fast":
        speed_cat_mapped = "moderate"
    else:
        speed_cat_mapped = speed_cat

    score = rc.calculate_risk_score(frame_data, speed_cat_mapped)
    model_pred = rm.predict_risk([desc])[0]

    print("DESCRIPTION:")
    print(desc)
    print()
    print("NUMERIC SCORE:", score)
    print("NUMERIC LABEL:", rc.score_to_label(score))
    print("MODEL PRED:", model_pred, "->", rm.interpret_risk(model_pred))


if __name__ == '__main__':
    # Example 1: frame that previously produced the danger tokens
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

    print("\n--- Diagnosis: ORIGINAL SUSPECT FRAME ---\n")
    diagnose(sample_frame_1)

    # Example 2: what it should be if wrong-side is NOT detected and TTC not critical
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

    print("\n--- Diagnosis: CORRECTED FRAME (no wrong-side, lower proximity) ---\n")
    diagnose(sample_frame_2)
