from core.config import PipelineConfig
from risk_calculator import RiskCalculator
from scoring.fusion import merge_labels
from scoring.smoothing import smooth_labels
from scoring.verdict import compute_verdict


def test_single_extreme_frame_triggers_danger_moment_caution():
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
    assert result.verdict == "CAUTION_WITH_DANGER_MOMENT"


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


def test_mostly_safe_ride_with_few_danger_frames_is_not_full_danger():
    predictions = [0] * 113
    numeric_scores = [5] * 113
    for index in (5, 18, 31, 44, 57, 70, 83, 96, 109):
        predictions[index] = 2
        numeric_scores[index] = 100
    for index in (2, 9, 15, 22, 28, 35, 41, 48, 54, 61, 67, 74, 80, 87, 93, 100, 106):
        predictions[index] = 1
        numeric_scores[index] = 35
    result = compute_verdict(
        smoothed_predictions=predictions,
        numeric_scores=numeric_scores,
        raw_frame_data=[{} for _ in predictions],
        stats={},
        total_samples=len(predictions),
        config=PipelineConfig(),
    )
    assert result.verdict == "CAUTION_WITH_DANGER_MOMENT"


def test_traffic_jam_with_pedestrian_is_not_danger():
    rc = RiskCalculator()
    frame = {
        "pedestrian_crossing_risk": True,
        "ego_speed": "slow",
        "ttc_status": "stable",
        "proximity_score": 0.45,
        "traffic_jam": True,
        "front_stable_seconds": 3.0,
        "phone_risk": "safe",
    }
    score = rc.calculate_risk_score(frame, ego_speed_category="slow")
    assert rc.score_to_label(score) in ("SAFE", "CAUTION")


def test_close_stable_traffic_jam_is_safe_not_tailgating():
    rc = RiskCalculator()
    frame = {
        "ego_speed": "slow",
        "ttc_status": "stable",
        "proximity_score": 0.65,
        "front_ttc_seconds": None,
        "front_relative_speed_proxy": 0.0,
        "front_stable_seconds": 4.0,
        "traffic_jam": True,
        "objects": ["car", "bus", "motorcycle", "person"],
    }
    score = rc.calculate_risk_score(frame, "slow")
    assert rc.score_to_label(score) == "SAFE"


def test_low_ttc_in_forward_path_is_danger_even_without_raw_distance():
    rc = RiskCalculator()
    frame = {
        "ego_speed": "fast",
        "ttc_status": "critical_approach",
        "proximity_score": 0.28,
        "front_ttc_seconds": 1.2,
        "front_relative_speed_proxy": 0.14,
        "traffic_jam": False,
        "objects": ["car"],
    }
    score = rc.calculate_risk_score(frame, "fast")
    assert rc.score_to_label(score) == "DANGER"


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
