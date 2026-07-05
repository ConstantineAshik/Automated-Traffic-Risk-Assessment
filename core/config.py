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
    structured_risk_model_path: Optional[str] = None
    use_mock_risk_model: bool = False
    danger_pct_threshold: float = 0.03
    caution_pct_threshold: float = 0.20
    max_score_danger: int = 95
    max_score_moderate: int = 80
    max_run_danger: int = 3
    episode_window_len: int = 10
