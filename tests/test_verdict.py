import sys
import os
import pytest
from core.config import PipelineConfig
from scoring.fusion import merge_labels
from scoring.smoothing import smooth_labels
from scoring.verdict import compute_verdict

# Ensure project root is on sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from risk_calculator import RiskCalculator


def test_single_extreme_frame_triggers_danger():
    rc = RiskCalculator()
    # synthetic sequence: one extreme frame in the middle
    model_preds = [0, 0, 0]
    numeric_scores = [10, 100, 10]
    raw_frames = [ {}, {}, {} ]

    numeric_labels = [rc.score_to_label(s) for s in numeric_scores]
    merged = merge_labels(model_preds, numeric_labels)
    sm = smooth_labels(merged, window=5)
    verdict = compute_verdict(sm, numeric_scores, raw_frames, {}, len(sm), PipelineConfig()).verdict
    assert verdict == "DANGER"


def test_three_consecutive_danger_frames():
    rc = RiskCalculator()
    # three consecutive high-risk frames
    model_preds = [0, 0, 0, 0, 0]
    numeric_scores = [10, 100, 100, 100, 10]
    raw_frames = [ {}, {}, {}, {}, {} ]

    numeric_labels = [rc.score_to_label(s) for s in numeric_scores]
    merged = merge_labels(model_preds, numeric_labels)
    sm = smooth_labels(merged, window=5)
    verdict = compute_verdict(sm, numeric_scores, raw_frames, {}, len(sm), PipelineConfig()).verdict
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
