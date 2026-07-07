import argparse
import json
from pathlib import Path

from ultralytics import YOLO


COCO_TRAFFIC_LABELS = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "cell phone",
}
DOMAIN_LABELS = {"rickshaw", "cng", "auto rickshaw", "wrong way vehicle"}


def inspect(path: Path):
    model = YOLO(str(path))
    names = model.names
    labels = list(names.values()) if isinstance(names, dict) else list(names)
    normalized = {str(label).lower().replace("_", " ") for label in labels}
    args = getattr(model.model, "args", {}) or {}
    return {
        "path": str(path),
        "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
        "task": model.task,
        "class_count": len(labels),
        "dataset_metadata": args.get("data"),
        "traffic_labels": sorted(normalized & COCO_TRAFFIC_LABELS),
        "domain_labels": sorted(normalized & DOMAIN_LABELS),
        "appears_domain_specific": bool(normalized & DOMAIN_LABELS),
    }


def main():
    parser = argparse.ArgumentParser(description="Inspect YOLO checkpoint metadata.")
    parser.add_argument("models", nargs="*", type=Path)
    args = parser.parse_args()
    paths = args.models or sorted(Path("models").glob("*.pt"))
    results = []
    for path in paths:
        try:
            results.append(inspect(path))
        except Exception as exc:
            results.append(
                {
                    "path": str(path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
