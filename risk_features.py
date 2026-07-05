from typing import Dict, Iterable, List


CATEGORICAL_FEATURES = ["ego_speed", "ttc_status"]
NUMERIC_FEATURES = [
    "proximity_score",
    "object_count",
    "heavy_vehicle_count",
    "pedestrian_count",
    "is_erratic",
    "glare",
    "night",
    "wet_or_glare_surface",
    "phone_detected",
    "phone_mounted",
    "side_cut_risk",
    "wrong_side_risk",
    "frontal_conflict_risk",
    "short_follow_distance",
    "pinch_point",
    "bus_blind_spot",
    "entering_traffic_risk",
    "pedestrian_crossing_risk",
    "late_night_high_speed",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def frame_to_features(frame_data: Dict) -> Dict:
    objects = frame_data.get("objects", [])
    return {
        "ego_speed": str(frame_data.get("ego_speed", "unknown")),
        "ttc_status": str(frame_data.get("ttc_status", "unknown")),
        "proximity_score": float(frame_data.get("proximity_score", 0.0)),
        "object_count": len(objects),
        "heavy_vehicle_count": sum(
            obj in ("bus", "truck", "heavy_vehicle") for obj in objects
        ),
        "pedestrian_count": sum(obj == "person" for obj in objects),
        **{
            name: int(bool(frame_data.get(name, False)))
            for name in NUMERIC_FEATURES
            if name
            not in {
                "proximity_score",
                "object_count",
                "heavy_vehicle_count",
                "pedestrian_count",
            }
        },
    }


def frames_to_features(frames: Iterable[Dict]) -> List[Dict]:
    return [frame_to_features(frame) for frame in frames]
