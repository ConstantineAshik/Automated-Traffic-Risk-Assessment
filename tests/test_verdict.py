from core.config import PipelineConfig
from risk_calculator import RiskCalculator
from scoring.fusion import merge_labels
from scoring.smoothing import smooth_labels
from scoring.verdict import compute_verdict


def test_single_extreme_frame_triggers_danger():
    rc = RiskCalculator()
    model_predictions = [0, 0, 0]
    numeric_scores = [10, 100, 10]
    raw_frames = [{}, {}, {}]

    numeric_labels = [rc.score_to_label(score) for score in numeric_scores]
    merged = merge_labels(model_predictions, numeric_labels)
    smoothed = smooth_labels(merged, window_size=3)
    result = compute_verdict(
        smoothed,
        numeric_scores,
        raw_frames,
        {},
        len(smoothed),
        PipelineConfig(),
    )
    assert result.verdict == "DANGER"


def test_three_consecutive_danger_frames():
    rc = RiskCalculator()
    model_predictions = [0, 2, 2, 2, 0]
    numeric_scores = [10, 70, 70, 70, 10]
    raw_frames = [{}, {}, {}, {}, {}]

    numeric_labels = [rc.score_to_label(score) for score in numeric_scores]
    merged = merge_labels(model_predictions, numeric_labels)
    smoothed = smooth_labels(merged, window_size=3)
    result = compute_verdict(
        smoothed,
        numeric_scores,
        raw_frames,
        {},
        len(smoothed),
        PipelineConfig(),
    )
    assert result.verdict == "DANGER"
    assert result.max_run >= 3


def test_traffic_jam_with_pedestrian_is_not_danger():
    rc = RiskCalculator()
    frame = {
        "pedestrian_crossing_risk": True,
        "ego_speed": "slow",
        "ttc_status": "stable",
        "proximity_score": 0.45,
        "phone_risk": "safe",
    }
    score = rc.calculate_risk_score(frame, ego_speed_category="slow")
    assert rc.score_to_label(score) in ("SAFE", "CAUTION")


def test_fast_branch_is_more_conservative_than_moderate():
    rc = RiskCalculator()
    frame = {
        "proximity_score": 0.55,
        "ttc_status": "critical_approach",
        "objects": ["truck"],
        "phone_risk": "safe",
    }
    moderate = rc.calculate_risk_score(frame, "moderate")
    fast = rc.calculate_risk_score(frame, "fast")
    assert fast > moderate
    assert rc.score_to_label(fast) == "DANGER"


def test_smoothing_does_not_expand_one_danger_into_a_run():
    raw = [0, 0, 2, 0, 0, 0, 0]
    smoothed = smooth_labels(raw, window_size=5)
    assert smoothed.count(2) <= raw.count(2)


def test_smoothing_rejects_invalid_window():
    try:
        smooth_labels([0, 1], window_size=0)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for a zero-sized smoothing window")


def test_two_class_disagreement_uses_caution():
    assert merge_labels([2], ["SAFE"]) == [1]
    assert merge_labels([0], ["DANGER"]) == [1]


def test_short_all_safe_video_remains_safe():
    result = compute_verdict(
        smoothed_predictions=[0],
        numeric_scores=[0],
        raw_frame_data=[{}],
        stats={},
        total_samples=1,
        config=PipelineConfig(),
    )
    assert result.episode_count == 0
    assert result.verdict == "SAFE"
