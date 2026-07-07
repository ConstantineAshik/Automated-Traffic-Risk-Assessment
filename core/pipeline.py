from typing import Dict, List

from core.config import PipelineConfig
from core.types import AnalysisResult
from scoring.fusion import merge_labels
from scoring.smoothing import smooth_labels
from scoring.verdict import compute_verdict
from video_processor import VideoProcessor
from text_generator import TextGenerator
from risk_calculator import RiskCalculator


def _map_speed_category(speed_status: str) -> str:
    if speed_status == "stationary":
        return "stationary"
    if speed_status == "slow":
        return "slow"
    if speed_status == "fast":
        return "fast"
    return "slow"


def _init_stats() -> Dict[str, int]:
    return {
        "High Speed Tailgating": 0,
        "Phone Distraction (sustained)": 0,
        "Phone Usage Caution (brief)": 0,
        "Glare Blindness": 0,
        "Side Cut Risk": 0,
        "Wrong Side Risk": 0,
        "Frontal Conflict": 0,
        "Short Follow Distance": 0,
        "Pinch Point": 0,
        "Bus Blind Spot": 0,
        "Unsecured Load": 0,
        "Entering Traffic Conflict": 0,
        "Pedestrian Crossing": 0,
        "Wet Road / Glare": 0,
        "Late-Night High-Speed": 0,
        "Traffic Jam": 0,
        "Low TTC Approach": 0,
        "Stable Close Distance": 0,
    }


def _update_stats(stats: Dict[str, int], desc: str, frame_data: Dict) -> None:
    phone_risk = frame_data.get("phone_risk", "safe")
    if phone_risk == "danger":
        stats["Phone Distraction (sustained)"] += 1
    elif phone_risk == "caution":
        stats["Phone Usage Caution (brief)"] += 1

    if "high_speed_tailgating" in desc:
        stats["High Speed Tailgating"] += 1
    if frame_data.get("glare", False):
        stats["Glare Blindness"] += 1
    if frame_data.get("side_cut_risk", False):
        stats["Side Cut Risk"] += 1
    if frame_data.get("wrong_side_risk", False):
        stats["Wrong Side Risk"] += 1
    if frame_data.get("frontal_conflict_risk", False):
        stats["Frontal Conflict"] += 1
    if frame_data.get("short_follow_distance", False):
        stats["Short Follow Distance"] += 1
    if frame_data.get("pinch_point", False):
        stats["Pinch Point"] += 1
    if frame_data.get("bus_blind_spot", False):
        stats["Bus Blind Spot"] += 1
    if frame_data.get("unsecured_load_risk", False):
        stats["Unsecured Load"] += 1
    if frame_data.get("entering_traffic_risk", False):
        stats["Entering Traffic Conflict"] += 1
    if frame_data.get("pedestrian_crossing_risk", False):
        stats["Pedestrian Crossing"] += 1
    if frame_data.get("wet_or_glare_surface", False):
        stats["Wet Road / Glare"] += 1
    if frame_data.get("late_night_high_speed", False):
        stats["Late-Night High-Speed"] += 1
    if frame_data.get("traffic_jam", False):
        stats["Traffic Jam"] += 1
    front_ttc = frame_data.get("front_ttc_seconds")
    if front_ttc is not None and front_ttc <= 3.0:
        stats["Low TTC Approach"] += 1
    if frame_data.get("front_stable_seconds", 0.0) >= 2.0:
        stats["Stable Close Distance"] += 1


def analyze(video_path: str, config: PipelineConfig) -> AnalysisResult:
    processor = VideoProcessor(
        video_path,
        window_size=10,
        danger_img_dir="danger_frames",
        sampling_fps=config.sampling_fps,
        max_frames=config.max_frames,
        model_paths=config.model_paths,
        detection_confidence=config.detection_confidence,
        ensemble_iou_threshold=config.ensemble_iou_threshold,
        ensemble_min_model_votes=config.ensemble_min_model_votes,
        inference_image_size=config.inference_image_size,
        inference_device=config.inference_device,
        detection_dataset=config.detection_dataset,
        detect_all_coco_objects=config.detect_all_coco_objects,
    )
    text_gen = TextGenerator()
    risk_calc = RiskCalculator()

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

    numeric_scores: List[int] = []
    for frame_data in raw_frame_data:
        speed_cat = _map_speed_category(frame_data.get("ego_speed", "slow"))
        score = risk_calc.calculate_risk_score(frame_data, speed_cat)
        numeric_scores.append(score)

    numeric_labels = [risk_calc.score_to_label(score) for score in numeric_scores]
    label_values = {"SAFE": 0, "CAUTION": 1, "DANGER": 2}
    numeric_predictions = [label_values[label] for label in numeric_labels]

    if config.structured_risk_model_path:
        from structured_risk_model import StructuredRiskModel

        risk_model = StructuredRiskModel(config.structured_risk_model_path)
        model_predictions = risk_model.predict(raw_frame_data)
        merged_predictions = merge_labels(model_predictions, numeric_labels)
    elif config.use_mock_risk_model:
        # Explicit opt-in only: this model is useful for demonstrations, not for
        # claiming real-world accuracy because its examples are handcrafted.
        from risk_model import RiskModel

        risk_model = RiskModel()
        risk_model.train_mock_model()
        model_predictions = list(map(int, risk_model.predict_risk(descriptions)))
        merged_predictions = merge_labels(model_predictions, numeric_labels)
    else:
        model_predictions = numeric_predictions
        merged_predictions = numeric_predictions
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
        detector_metadata=processor.detector_metadata,
    )
