from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class PipelineConfig:
    sampling_fps: float = 2.0
    max_frames: int = 0
    smoothing_window: int = 3
    model_paths: Tuple[str, ...] = (
        "models/yolo11n.pt",
        "models/yolo11m.pt",
        "models/yolo12m.pt",
    )
    detection_confidence: float = 0.25
    ensemble_iou_threshold: float = 0.55
    ensemble_min_model_votes: int = 1
    inference_image_size: int = 640
    inference_device: Optional[str] = None
    detection_dataset: str = "coco"
    detect_all_coco_objects: bool = True
    structured_risk_model_path: Optional[str] = None
    use_mock_risk_model: bool = False
    danger_pct_threshold: float = 0.30
    caution_pct_threshold: float = 0.25
    max_score_danger: int = 90
    max_score_moderate: int = 80
    max_run_danger: int = 3
    average_score_caution: float = 35.0
    episode_window_len: int = 10
