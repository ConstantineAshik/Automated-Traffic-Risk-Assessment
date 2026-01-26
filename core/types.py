from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class AnalysisResult:
    video_path: str
    raw_frame_data: List[Dict]
    descriptions: List[str]
    model_predictions: List[int]
    numeric_scores: List[int]
    numeric_labels: List[str]
    merged_predictions: List[int]
    smoothed_predictions: List[int]
    stats: Dict[str, int]
    safe_count: int
    caution_count: int
    danger_count: int
    verdict: str
    verdict_reason: str
    failure_rate: float
    incomplete_analysis: bool
    max_score: int
    max_run: int
    episode_count: int
    phone_danger_frames: int
    total_samples: int
