from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ultralytics import YOLO


Box = Tuple[float, float, float, float]


LABEL_ALIASES = {
    "cellphone": "cell phone",
    "mobile phone": "cell phone",
    "motorbike": "motorcycle",
    "auto rickshaw": "rickshaw",
    "auto-rickshaw": "rickshaw",
    "cng": "rickshaw",
    "cng auto rickshaw": "rickshaw",
}

RELEVANT_LABELS = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "cell phone",
    "rickshaw",
    "unsecured load",
    "wrong way vehicle",
}

COCO_DATASET_NAME = "coco"


def normalize_label(label: str) -> str:
    normalized = " ".join(str(label).strip().lower().replace("_", " ").split())
    return LABEL_ALIASES.get(normalized, normalized)


def box_iou(first: Box, second: Box) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


@dataclass
class Detection:
    box: Box
    label: str
    confidence: float
    sources: Set[str] = field(default_factory=set)
    track_id: Optional[int] = None
    ttc_status: str = "unknown"
    ttc_seconds: Optional[float] = None
    distance_proxy: Optional[float] = None
    relative_speed_proxy: float = 0.0
    stable_seconds: float = 0.0
    lateral_velocity: float = 0.0
    in_forward_path: bool = False

    @property
    def model_votes(self) -> int:
        return len(self.sources)


def _weighted_box(detections: Iterable[Detection]) -> Box:
    items = list(detections)
    total_weight = sum(max(item.confidence, 0.01) for item in items)
    return tuple(
        sum(item.box[index] * max(item.confidence, 0.01) for item in items)
        / total_weight
        for index in range(4)
    )


def fuse_detections(
    detections: Sequence[Detection],
    iou_threshold: float,
    min_model_votes: int,
) -> List[Detection]:
    """Fuse same-class boxes while retaining the number of independent model votes."""
    clusters: List[List[Detection]] = []
    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        best_cluster = None
        best_iou = 0.0
        for cluster in clusters:
            if cluster[0].label != detection.label:
                continue
            overlap = box_iou(_weighted_box(cluster), detection.box)
            if overlap >= iou_threshold and overlap > best_iou:
                best_cluster = cluster
                best_iou = overlap
        if best_cluster is None:
            clusters.append([detection])
        else:
            best_cluster.append(detection)

    fused: List[Detection] = []
    for cluster in clusters:
        sources = set().union(*(item.sources for item in cluster))
        if len(sources) < min_model_votes:
            continue
        # Average confidence is deliberately not summed: three mediocre detections
        # should not become a falsely certain result.
        confidence = sum(item.confidence for item in cluster) / len(cluster)
        fused.append(
            Detection(
                box=_weighted_box(cluster),
                label=cluster[0].label,
                confidence=confidence,
                sources=sources,
            )
        )
    return sorted(fused, key=lambda item: item.confidence, reverse=True)


class EnsembleDetector:
    def __init__(
        self,
        model_paths: Sequence[str],
        confidence: float = 0.25,
        iou_threshold: float = 0.55,
        min_model_votes: int = 1,
        image_size: int = 640,
        device: Optional[str] = None,
        dataset: str = COCO_DATASET_NAME,
        detect_all_labels: bool = True,
    ):
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.min_model_votes = min_model_votes
        self.image_size = image_size
        self.device = device
        self.dataset = dataset
        self.detect_all_labels = detect_all_labels
        self.models = []
        self.load_errors: Dict[str, str] = {}

        for raw_path in model_paths:
            path = Path(raw_path)
            try:
                model = YOLO(str(path))
                self.models.append((path.name, model))
            except Exception as exc:
                self.load_errors[str(path)] = f"{type(exc).__name__}: {exc}"

        if not self.models:
            details = "; ".join(
                f"{path}: {error}" for path, error in self.load_errors.items()
            )
            raise RuntimeError(f"No object-detection model could be loaded. {details}")

    @property
    def model_names(self) -> List[str]:
        return [name for name, _ in self.models]

    def metadata(self) -> Dict:
        return {
            "loaded_models": self.model_names,
            "load_errors": dict(self.load_errors),
            "confidence_threshold": self.confidence,
            "ensemble_iou_threshold": self.iou_threshold,
            "minimum_model_votes": self.min_model_votes,
            "dataset": self.dataset,
            "detect_all_labels": self.detect_all_labels,
        }

    def predict(self, frame) -> Tuple[List[Detection], bool, Dict[str, str]]:
        candidates: List[Detection] = []
        inference_errors: Dict[str, str] = {}
        successful_models = 0

        for model_name, model in self.models:
            try:
                kwargs = {
                    "source": frame,
                    "verbose": False,
                    "conf": self.confidence,
                    "imgsz": self.image_size,
                }
                if self.device:
                    kwargs["device"] = self.device
                result = model.predict(**kwargs)[0]
                successful_models += 1
            except Exception as exc:
                inference_errors[model_name] = f"{type(exc).__name__}: {exc}"
                continue

            names = result.names
            for box in result.boxes:
                class_id = int(box.cls[0])
                raw_label = (
                    names.get(class_id, str(class_id))
                    if isinstance(names, dict)
                    else names[class_id]
                )
                label = normalize_label(raw_label)
                if not self.detect_all_labels and label not in RELEVANT_LABELS:
                    continue
                coordinates = tuple(float(value) for value in box.xyxy[0].tolist())
                candidates.append(
                    Detection(
                        box=coordinates,
                        label=label,
                        confidence=float(box.conf[0]),
                        sources={model_name},
                    )
                )

        fused = fuse_detections(
            candidates,
            iou_threshold=self.iou_threshold,
            min_model_votes=min(self.min_model_votes, max(successful_models, 1)),
        )
        return fused, successful_models > 0, inference_errors
