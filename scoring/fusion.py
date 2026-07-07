from typing import List


def merge_labels(model_labels: List[int], numeric_labels: List[str]) -> List[int]:
    """Merge two risk opinions without allowing either to jump two classes.

    This remains available for a future validated risk classifier. The current
    production pipeline uses the numeric risk score alone because the bundled
    text classifier is trained only on mock examples.
    """
    numeric_mapping = {"SAFE": 0, "CAUTION": 1, "DANGER": 2}
    merged = []
    for model_label, numeric_label in zip(model_labels, numeric_labels):
        model_value = int(model_label)
        numeric_value = numeric_mapping[numeric_label]
        if abs(model_value - numeric_value) == 2:
            merged.append(1)
        else:
            merged.append(max(model_value, numeric_value))
    return merged
