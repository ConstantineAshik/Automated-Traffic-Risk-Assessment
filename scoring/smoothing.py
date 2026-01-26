from collections import deque
from typing import List


def smooth_labels(labels: List[int], window_size: int) -> List[int]:
    risk_window = deque(maxlen=window_size)
    smoothed = []
    for lab in labels:
        risk_window.append(int(lab))
        smoothed.append(max(risk_window))
    return smoothed
