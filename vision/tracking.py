import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Sequence, Tuple

from vision.ensemble import Box, Detection, box_iou


@dataclass
class Track:
    track_id: int
    label: str
    box: Box
    last_seen: float
    history: Deque[Tuple[float, Box]] = field(default_factory=lambda: deque(maxlen=6))


def _center(box: Box) -> Tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


class IoUTracker:
    """Small dependency-free tracker suitable for sparsely sampled inference frames."""

    def __init__(self, minimum_iou: float = 0.10, max_age_seconds: float = 2.0):
        self.minimum_iou = minimum_iou
        self.max_age_seconds = max_age_seconds
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1

    def _similarity(
        self, detection: Detection, track: Track, frame_width: float
    ) -> float:
        overlap = box_iou(detection.box, track.box)
        detection_center, _ = _center(detection.box)
        track_center, _ = _center(track.box)
        center_distance = abs(detection_center - track_center) / max(frame_width, 1.0)
        if overlap < self.minimum_iou and center_distance > 0.12:
            return -1.0
        return overlap - (0.25 * center_distance)

    def update(
        self,
        detections: Sequence[Detection],
        timestamp: float,
        frame_width: int,
        frame_height: int,
    ) -> List[Detection]:
        self.tracks = {
            track_id: track
            for track_id, track in self.tracks.items()
            if timestamp - track.last_seen <= self.max_age_seconds
        }
        available_track_ids = set(self.tracks)

        for detection in detections:
            best_track_id = None
            best_score = -1.0
            for track_id in available_track_ids:
                track = self.tracks[track_id]
                if track.label != detection.label:
                    continue
                score = self._similarity(detection, track, frame_width)
                if score > best_score:
                    best_track_id = track_id
                    best_score = score

            if best_track_id is None:
                best_track_id = self.next_track_id
                self.next_track_id += 1
                track = Track(
                    track_id=best_track_id,
                    label=detection.label,
                    box=detection.box,
                    last_seen=timestamp,
                )
                track.history.append((timestamp, detection.box))
                self.tracks[best_track_id] = track
            else:
                track = self.tracks[best_track_id]
                available_track_ids.remove(best_track_id)
                track.box = detection.box
                track.last_seen = timestamp
                track.history.append((timestamp, detection.box))

            detection.track_id = best_track_id
            self._add_motion_metrics(detection, track, frame_width, frame_height)

        return list(detections)

    @staticmethod
    def _add_motion_metrics(
        detection: Detection, track: Track, frame_width: int, frame_height: int
    ) -> None:
        detection.distance_proxy = _distance_proxy(detection.box, frame_width, frame_height)
        if len(track.history) < 2:
            return
        old_time, old_box = track.history[0]
        new_time, new_box = track.history[-1]
        elapsed = new_time - old_time
        if elapsed <= 0:
            return

        old_distance = _distance_proxy(old_box, frame_width, frame_height)
        new_distance = _distance_proxy(new_box, frame_width, frame_height)
        # Positive relative speed means the object is getting closer in the
        # monocular-camera proxy space. Near zero means traffic is moving with
        # the rider, or both are stopped, so TTC should not create danger.
        relative_speed = (old_distance - new_distance) / elapsed
        detection.distance_proxy = new_distance
        detection.relative_speed_proxy = relative_speed

        if relative_speed > 0.03:
            ttc_seconds = new_distance / relative_speed
            detection.ttc_seconds = ttc_seconds
            if ttc_seconds < 1.5:
                detection.ttc_status = "critical_approach"
            elif ttc_seconds <= 3.0:
                detection.ttc_status = "closing_in"
            else:
                detection.ttc_status = "stable"
        else:
            detection.ttc_seconds = math.inf
            detection.ttc_status = "stable"

        old_center, _ = _center(old_box)
        new_center, _ = _center(new_box)
        detection.lateral_velocity = (
            (new_center - old_center) / max(frame_width, 1)
        ) / elapsed

        close_now = _closeness(new_box, frame_width, frame_height) > 0.32
        if close_now and abs(relative_speed) <= 0.03:
            detection.stable_seconds = elapsed
        else:
            detection.stable_seconds = 0.0


def _closeness(box: Box, frame_width: int, frame_height: int) -> float:
    width_closeness = (box[2] - box[0]) / (max(frame_width, 1) * 0.8)
    height_closeness = (box[3] - box[1]) / (max(frame_height, 1) * 0.9)
    return min(max(width_closeness, height_closeness), 1.0)


def _distance_proxy(box: Box, frame_width: int, frame_height: int) -> float:
    closeness = max(_closeness(box, frame_width, frame_height), 0.02)
    return 1.0 / closeness
