from collections import Counter, deque
from typing import List


def smooth_labels(labels: List[int], window_size: int) -> List[int]:
    """Causal majority smoothing that never invents a sustained danger run.

    Ties are resolved in favor of the newest observation, which keeps the
    output responsive without copying a single maximum across the full window.
    """
    if window_size < 1:
        raise ValueError("window_size must be at least 1")

    risk_window = deque(maxlen=window_size)
    smoothed = []
    for lab in labels:
        risk_window.append(int(lab))
        counts = Counter(risk_window)
        largest_count = max(counts.values())
        tied = {label for label, count in counts.items() if count == largest_count}
        newest_tied_label = next(
            label for label in reversed(risk_window) if label in tied
        )
        smoothed.append(newest_tied_label)
    return smoothed
