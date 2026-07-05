import csv
import os
from typing import Optional

from core.types import AnalysisResult
from risk_features import FEATURE_COLUMNS, frame_to_features


def write_predictions_csv(result: AnalysisResult, output_path: Optional[str] = None) -> str:
    if output_path is None:
        video_stem = os.path.splitext(os.path.basename(result.video_path))[0]
        output_path = os.path.join(
            os.getcwd(), f"{video_stem}_predictions.csv"
        )

    with open(output_path, "w", newline="", encoding="utf-8") as cf:
        fieldnames = [
            "ride_id",
            "video_path",
            "frame_id",
            "timestamp_seconds",
            "model_label",
            "numeric_label",
            "merged_label",
            "description",
            "numeric_score",
            "phone_risk",
            "smoothed_label",
            *FEATURE_COLUMNS,
            "human_label",
        ]
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        ride_id = os.path.splitext(os.path.basename(result.video_path))[0]
        for i, (desc, model_lab, nlabel, merged, score) in enumerate(
            zip(
                result.descriptions,
                result.model_predictions,
                result.numeric_labels,
                result.merged_predictions,
                result.numeric_scores,
            )
        ):
            frame_id = result.raw_frame_data[i].get("frame_id", i)
            phone_risk = result.raw_frame_data[i].get("phone_risk", "")
            row = {
                "ride_id": ride_id,
                "video_path": result.video_path,
                "frame_id": int(frame_id),
                "timestamp_seconds": result.raw_frame_data[i].get("timestamp", ""),
                "model_label": int(model_lab),
                "numeric_label": nlabel,
                "merged_label": int(merged),
                "description": desc,
                "numeric_score": f"{score:.2f}",
                "phone_risk": phone_risk,
                "smoothed_label": int(result.smoothed_predictions[i]),
                "human_label": "",
            }
            row.update(frame_to_features(result.raw_frame_data[i]))
            writer.writerow(row)
    return output_path
