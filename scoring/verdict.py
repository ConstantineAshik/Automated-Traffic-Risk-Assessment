from dataclasses import dataclass
from typing import Dict, List

from core.config import PipelineConfig


@dataclass
class VerdictResult:
    verdict: str
    reason: str
    max_score: int
    max_run: int
    episode_count: int
    phone_danger_frames: int


def _count_max_run(smoothed_predictions: List[int]) -> int:
    max_run = 0
    current_run = 0
    for r in smoothed_predictions:
        if r == 2:
            current_run += 1
            if current_run > max_run:
                max_run = current_run
        else:
            current_run = 0
    return max_run


def _count_episodes(smoothed_predictions: List[int], window_len: int) -> int:
    if window_len < 1:
        raise ValueError("episode window length must be at least 1")
    episode_count = 0
    for i in range(0, len(smoothed_predictions), window_len):
        window = smoothed_predictions[i : i + window_len]
        if not window:
            continue
        required_danger_frames = (len(window) // 2) + 1
        if sum(1 for r in window if r == 2) >= required_danger_frames:
            episode_count += 1
    return episode_count


def compute_verdict(
    smoothed_predictions: List[int],
    numeric_scores: List[int],
    raw_frame_data: List[Dict],
    stats: Dict[str, int],
    total_samples: int,
    config: PipelineConfig,
) -> VerdictResult:
    danger_count = sum(1 for r in smoothed_predictions if r == 2)
    caution_count = sum(1 for r in smoothed_predictions if r == 1)
    danger_ratio = (danger_count / total_samples) if total_samples > 0 else 0
    caution_ratio = (caution_count / total_samples) if total_samples > 0 else 0
    danger_pct = danger_ratio * 100
    caution_pct = caution_ratio * 100

    max_score = max(numeric_scores) if numeric_scores else 0
    average_score = (
        sum(numeric_scores) / len(numeric_scores) if numeric_scores else 0
    )
    max_run = _count_max_run(smoothed_predictions)
    episode_count = _count_episodes(smoothed_predictions, config.episode_window_len)
    phone_danger_frames = sum(1 for f in raw_frame_data if f.get("phone_risk") == "danger")

    verdict = "SAFE"
    reason = f"Good riding with {100 - caution_pct - danger_pct:.1f}% safe frames."

    if (
        phone_danger_frames > 0
        or stats.get("Phone Distraction (sustained)", 0) > 0
    ):
        verdict = "DANGER"
        reason = "Sustained phone distraction detected. Immediate corrective action required."
    elif (
        danger_ratio >= config.danger_pct_threshold
        or max_run >= config.max_run_danger
    ):
        verdict = "DANGER"
        reason = (
            "Danger was frequent or sustained "
            f"({danger_pct:.1f}% danger, longest run {max_run})."
        )
    elif max_score >= config.max_score_danger:
        verdict = "CAUTION_WITH_DANGER_MOMENT"
        reason = (
            "Mostly acceptable riding, but at least one high-risk moment "
            f"reached {max_score}/100."
        )
    elif (
        danger_ratio > 0
        or caution_ratio >= config.caution_pct_threshold
        or average_score >= config.average_score_caution
        or episode_count >= 1
        or max_score >= config.max_score_moderate
    ):
        verdict = "CAUTION"
        reason = (
            "Noticeable risky behavior detected without sustained danger "
            f"({caution_pct:.1f}% caution, average score {average_score:.1f}/100)."
        )

    if verdict == "SAFE":
        if stats.get("Wrong Side Risk", 0) > 0 and max_score >= 70:
            verdict = "CAUTION"
            reason = "Wrong-side interactions detected alongside high scores."
        if stats.get("Pedestrian Crossing", 0) > 0 and max_score >= 65:
            verdict = "CAUTION"
            reason = "Pedestrian interactions with high approach scores."

    return VerdictResult(
        verdict=verdict,
        reason=reason,
        max_score=max_score,
        max_run=max_run,
        episode_count=episode_count,
        phone_danger_frames=phone_danger_frames,
    )
