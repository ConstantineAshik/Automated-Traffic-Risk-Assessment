from typing import List


def merge_labels(model_labels: List[int], numeric_labels: List[str]) -> List[int]:
    merged = []
    for model_label, numeric_label in zip(model_labels, numeric_labels):
        if numeric_label == "DANGER":
            merged.append(2)
            continue
        if numeric_label == "SAFE" and int(model_label) == 0:
            merged.append(0)
            continue
        if int(model_label) == 2 or numeric_label == "CAUTION":
            merged.append(1)
            continue
        merged.append(int(model_label))
    return merged
