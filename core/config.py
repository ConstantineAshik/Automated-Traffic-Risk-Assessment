from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    sampling_fps: float = 2.0
    max_frames: int = 0
    smoothing_window: int = 5
    danger_pct_threshold: float = 0.03
    caution_pct_threshold: float = 0.20
    max_score_danger: int = 95
    max_score_moderate: int = 80
    max_run_danger: int = 3
    episode_window_len: int = 10
