import csv
import os
from typing import Optional

from core.types import AnalysisResult


def write_predictions_csv(result: AnalysisResult, output_path: Optional[str] = None) -> str:
    if output_path is None:
        output_path = os.path.join(os.getcwd(), "model_predictions.csv")

    with open(output_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow(
            [
                "frame_id",
                "model_label",
                "numeric_label",
                "merged_label",
                "description",
                "numeric_score",
                "phone_risk",
                "smoothed_label",
            ]
        )
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
            writer.writerow(
                [
                    int(frame_id),
                    int(model_lab),
                    nlabel,
                    int(merged),
                    desc,
                    f"{score:.2f}",
                    phone_risk,
                    int(result.smoothed_predictions[i]),
                ]
            )
    return output_path
