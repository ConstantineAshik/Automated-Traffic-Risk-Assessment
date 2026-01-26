from typing import Dict, List

from core.config import PipelineConfig
from core.types import AnalysisResult
from scoring.fusion import merge_labels
from scoring.smoothing import smooth_labels
from scoring.verdict import compute_verdict
from video_processor import VideoProcessor
from text_generator import TextGenerator
from risk_model import RiskModel
from risk_calculator import RiskCalculator


def _map_speed_category(speed_status: str) -> str:
    if speed_status == "stationary":
        return "stationary"
    if speed_status == "slow":
        return "slow"
    if speed_status == "fast":
        return "moderate"
    return "slow"


def _init_stats() -> Dict[str, int]:
    return {
        "High Speed Tailgating": 0,
        "Phone Distraction (5+ frames)": 0,
        "Phone Usage Caution (brief)": 0,
        "Glare Blindness": 0,
        "Side Cut Risk": 0,
        "Wrong Side Risk": 0,
        "Short Follow Distance": 0,
        "Pinch Point": 0,
        "Bus Blind Spot": 0,
        "Unsecured Load": 0,
        "Entering Traffic Conflict": 0,
        "Pedestrian Crossing": 0,
        "Wet Road / Glare": 0,
        "Late-Night High-Speed": 0,
    }


def _update_stats(stats: Dict[str, int], desc: str, frame_data: Dict) -> None:
    phone_risk = frame_data.get("phone_risk", "safe")
    if phone_risk == "danger":
        stats["Phone Distraction (5+ frames)"] += 1
    elif phone_risk == "caution":
        stats["Phone Usage Caution (brief)"] += 1

    if "high_speed_tailgating" in desc:
        stats["High Speed Tailgating"] += 1
    if "glare_blindness" in desc:
        stats["Glare Blindness"] += 1
    if "side_cut_risk" in desc:
        stats["Side Cut Risk"] += 1
    if "wrong_side_risk" in desc:
        stats["Wrong Side Risk"] += 1
    if "short_follow_distance" in desc:
        stats["Short Follow Distance"] += 1
    if "pinch_point" in desc:
        stats["Pinch Point"] += 1
    if "bus_blind_spot" in desc:
        stats["Bus Blind Spot"] += 1
    if "unsecured_truck_load" in desc:
        stats["Unsecured Load"] += 1
    if "entering_traffic_conflict" in desc:
        stats["Entering Traffic Conflict"] += 1
    if "pedestrian_crossing" in desc:
        stats["Pedestrian Crossing"] += 1
    if "wet_road_glare" in desc:
        stats["Wet Road / Glare"] += 1
    if "late_night_high_speed" in desc:
        stats["Late-Night High-Speed"] += 1


def analyze(video_path: str, config: PipelineConfig) -> AnalysisResult:
    processor = VideoProcessor(
        video_path,
        window_size=10,
        danger_img_dir="danger_frames",
        sampling_fps=config.sampling_fps,
        max_frames=config.max_frames,
    )
    text_gen = TextGenerator()
    risk_model = RiskModel()
    risk_calc = RiskCalculator()

    risk_model.train_mock_model()

    raw_frame_data = processor.process_video()
    if not raw_frame_data:
        raise ValueError("No frames found in video.")

    failure_rate = (
        (processor.detection_failures / processor.total_frames_processed * 100)
        if processor.total_frames_processed > 0
        else 0
    )
    incomplete_analysis = failure_rate > 5

    descriptions = [text_gen.generate_description(f) for f in raw_frame_data]
    model_predictions = list(map(int, risk_model.predict_risk(descriptions)))

    numeric_scores: List[int] = []
    for frame_data in raw_frame_data:
        speed_cat = _map_speed_category(frame_data.get("ego_speed", "slow"))
        score = risk_calc.calculate_risk_score(frame_data, speed_cat)
        numeric_scores.append(score)

    numeric_labels = [risk_calc.score_to_label(score) for score in numeric_scores]
    merged_predictions = merge_labels(model_predictions, numeric_labels)
    smoothed_predictions = smooth_labels(merged_predictions, config.smoothing_window)

    stats = _init_stats()
    safe_count = 0
    caution_count = 0
    danger_count = 0

    for desc, smoothed, frame_data in zip(descriptions, smoothed_predictions, raw_frame_data):
        _update_stats(stats, desc, frame_data)
        if smoothed == 0:
            safe_count += 1
        elif smoothed == 1:
            caution_count += 1
        else:
            danger_count += 1

    total_samples = len(descriptions)
    verdict_result = compute_verdict(
        smoothed_predictions,
        numeric_scores,
        raw_frame_data,
        stats,
        total_samples,
        config,
    )

    return AnalysisResult(
        video_path=video_path,
        raw_frame_data=raw_frame_data,
        descriptions=descriptions,
        model_predictions=model_predictions,
        numeric_scores=numeric_scores,
        numeric_labels=numeric_labels,
        merged_predictions=merged_predictions,
        smoothed_predictions=smoothed_predictions,
        stats=stats,
        safe_count=safe_count,
        caution_count=caution_count,
        danger_count=danger_count,
        verdict=verdict_result.verdict,
        verdict_reason=verdict_result.reason,
        failure_rate=failure_rate,
        incomplete_analysis=incomplete_analysis,
        max_score=verdict_result.max_score,
        max_run=verdict_result.max_run,
        episode_count=verdict_result.episode_count,
        phone_danger_frames=verdict_result.phone_danger_frames,
        total_samples=total_samples,
    )
