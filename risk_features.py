from typing import Dict, Iterable, List


CATEGORICAL_FEATURES = ["ego_speed", "ttc_status"]
NUMERIC_FEATURES = [
    "proximity_score",
    "side_proximity_score",
    "front_ttc_seconds",
    "front_relative_speed_proxy",
    "front_stable_seconds",
    "traffic_jam",
    "traffic_density",
    "scene_motion",
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
DEFAULT_FEATURE_VALUES = {
    "ego_speed": "unknown",
    "ttc_status": "stable",
    "proximity_score": 0.0,
    "side_proximity_score": 0.0,
    "front_ttc_seconds": 999.0,
    "front_relative_speed_proxy": 0.0,
    "front_stable_seconds": 0.0,
    "traffic_jam": 0,
    "traffic_density": 0,
    "scene_motion": 0.0,
    "object_count": 0,
    "heavy_vehicle_count": 0,
    "pedestrian_count": 0,
    **{
        name: 0
        for name in NUMERIC_FEATURES
        if name
        not in {
            "proximity_score",
            "side_proximity_score",
            "front_ttc_seconds",
            "front_relative_speed_proxy",
            "front_stable_seconds",
            "traffic_jam",
            "traffic_density",
            "scene_motion",
            "object_count",
            "heavy_vehicle_count",
            "pedestrian_count",
        }
    },
}


def frame_to_features(frame_data: Dict) -> Dict:
    objects = frame_data.get("objects", [])
    front_ttc = frame_data.get("front_ttc_seconds")
    if front_ttc is None:
        front_ttc = 999.0
    return {
        "ego_speed": str(frame_data.get("ego_speed", "unknown")),
        "ttc_status": str(frame_data.get("ttc_status", "unknown")),
        "proximity_score": float(frame_data.get("proximity_score", 0.0)),
        "side_proximity_score": float(frame_data.get("side_proximity_score", 0.0)),
        "front_ttc_seconds": float(front_ttc),
        "front_relative_speed_proxy": float(
            frame_data.get("front_relative_speed_proxy", 0.0)
        ),
        "front_stable_seconds": float(frame_data.get("front_stable_seconds", 0.0)),
        "traffic_jam": int(bool(frame_data.get("traffic_jam", False))),
        "traffic_density": int(frame_data.get("traffic_density", 0)),
        "scene_motion": float(frame_data.get("scene_motion", 0.0)),
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
