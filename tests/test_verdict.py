import sys
import os
import pytest
from collections import deque

# Ensure project root is on sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from risk_calculator import RiskCalculator


def merge_label(model_label, numeric_score, risk_calc):
    numeric_label = risk_calc.score_to_label(numeric_score)
    if numeric_label == "DANGER":
        return 2
    if numeric_label == "SAFE" and int(model_label) == 0:
        return 0
    if int(model_label) == 2 or numeric_label == "CAUTION":
        return 1
    return int(model_label)


def smooth_labels(merged, window=5):
    q = deque(maxlen=window)
    out = []
    for l in merged:
        q.append(int(l))
        out.append(max(q))
    return out


def compute_verdict(smoothed_predictions, numeric_scores, raw_frame_data):
    total_samples = len(smoothed_predictions)
    danger_count = sum(1 for r in smoothed_predictions if r == 2)
    max_score = max(numeric_scores) if numeric_scores else 0

    # compute longest continuous danger run
    max_run = 0
    current_run = 0
    for r in smoothed_predictions:
        if r == 2:
            current_run += 1
            if current_run > max_run:
                max_run = current_run
        else:
            current_run = 0

    phone_danger_frames = sum(1 for f in raw_frame_data if f.get("phone_risk") == "danger")

    # Dhaka-tuned rules
    if phone_danger_frames > 0:
        return "UNSAFE"
    if max_score >= 95 or max_run >= 3:
        return "DANGER"
    if (danger_count / total_samples) > 0.03 or max_score >= 80:
        return "MODERATE"
    return "SAFE"


def test_single_extreme_frame_triggers_danger():
    rc = RiskCalculator()
    # synthetic sequence: one extreme frame in the middle
    model_preds = [0, 0, 0]
    numeric_scores = [10, 100, 10]
    raw_frames = [ {}, {}, {} ]

    merged = [merge_label(m, s, rc) for m, s in zip(model_preds, numeric_scores)]
    sm = smooth_labels(merged, window=5)
    verdict = compute_verdict(sm, numeric_scores, raw_frames)
    assert verdict == "DANGER"


def test_three_consecutive_danger_frames():
    rc = RiskCalculator()
    # three consecutive high-risk frames
    model_preds = [0, 0, 0, 0, 0]
    numeric_scores = [10, 100, 100, 100, 10]
    raw_frames = [ {}, {}, {}, {}, {} ]

    merged = [merge_label(m, s, rc) for m, s in zip(model_preds, numeric_scores)]
    sm = smooth_labels(merged, window=5)
    verdict = compute_verdict(sm, numeric_scores, raw_frames)
    assert verdict == "DANGER"


def test_traffic_jam_with_pedestrian_is_not_danger():
    rc = RiskCalculator()
    # simulate jam frames: slow speed, pedestrian present but not critical TTC
    frame = {
        "pedestrian_crossing_risk": True,
        "ego_speed": "slow",
        "ttc_status": "stable",
        "proximity_score": 0.45,
        "phone_risk": "safe",
    }
    # compute score
    score = rc.calculate_risk_score(frame, ego_speed_category="slow")
    label = rc.score_to_label(score)
    # In Dhaka tuning, jam+pedestrian without critical TTC should not be DANGER
    assert label in ("SAFE", "CAUTION")
