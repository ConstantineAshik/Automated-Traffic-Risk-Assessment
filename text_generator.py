class TextGenerator:
    def __init__(self):
        pass

    def estimate_speed_category(self, speed_status):
        if speed_status == "stationary":
            return "stationary"
        if speed_status == "slow":
            return "slow"
        if speed_status == "fast":
            return "fast"
        return "slow"

    def generate_description(self, frame_data):
        tokens = []

        speed = frame_data.get("ego_speed", "stationary")
        proximity = frame_data.get("proximity_score", 0.0)
        ttc_status = frame_data.get("ttc_status", "stable")
        objs = frame_data.get("objects", [])
        phone_risk = frame_data.get("phone_risk", "safe")

        has_heavy_obj = any(o in ("bus", "truck", "heavy_vehicle") for o in objs)
        has_pedestrian = "person" in objs

        if frame_data.get("glare", False):
            tokens.append("glare_blindness")
        elif frame_data.get("night", False):
            tokens.append("night")
            if frame_data.get("wet_or_glare_surface", False):
                tokens.append("wet_road_glare")
        else:
            tokens.append("day")

        if phone_risk == "danger":
            tokens.append("phone_distraction")
        elif phone_risk == "caution":
            tokens.append("phone_usage_caution")

        tokens.append(speed)

        is_traffic_jam = speed in ("slow", "stationary") and proximity > 0.3

        if is_traffic_jam:
            tokens.append("traffic_jam_proximity")
            if has_pedestrian:
                tokens.append("pedestrian_crossing")
            if frame_data.get("short_follow_distance", False):
                tokens.append("short_follow_distance")
            if frame_data.get("bus_blind_spot", False):
                tokens.append("bus_blind_spot")
            if proximity <= 0.3:
                tokens.append("safe_gap")
            elif proximity <= 0.5:
                tokens.append("moderate_distance")
            else:
                tokens.append("close_proximity")
            return " ".join(tokens)

        if speed == "fast":
            high_speed_support = (
                frame_data.get("is_erratic", False)
                or ttc_status == "critical_approach"
                or ttc_status == "closing_in"
                or proximity > 0.45
            )
            if high_speed_support:
                tokens.append("high_speed")

            if frame_data.get("wrong_side_risk", False):
                tokens.append("wrong_side_risk")
            if frame_data.get("side_cut_risk", False):
                tokens.append("side_cut_risk")
            if ttc_status == "critical_approach":
                tokens.append("rapid_closing_speed")

            if has_pedestrian and (
                ttc_status == "critical_approach" or frame_data.get("is_erratic", False)
            ):
                tokens.append("pedestrian_crossing")

            if frame_data.get("sandwich_risk", False):
                tokens.append("high_speed_sandwich_risk")
            if frame_data.get("pinch_point", False):
                tokens.append("pinch_point")

            if proximity > 0.4 and has_heavy_obj:
                tokens.append("high_speed_tailgating")

            if frame_data.get("night", False) and len(objs) <= 1 and high_speed_support:
                tokens.append("late_night_high_speed")
        else:
            if has_pedestrian and proximity > 0.25:
                tokens.append("pedestrian_crossing")
            if frame_data.get("short_follow_distance", False):
                tokens.append("short_follow_distance")
            if frame_data.get("pinch_point", False):
                tokens.append("pinch_point_no_escape")
            if frame_data.get("bus_blind_spot", False) and proximity > 0.3:
                tokens.append("bus_blind_spot")
            if frame_data.get("entering_traffic_risk", False):
                tokens.append("entering_traffic_conflict")

        if proximity > 0.5:
            tokens.append("close_proximity")
        elif proximity > 0.3:
            tokens.append("moderate_distance")
        else:
            tokens.append("safe_gap")

        return " ".join(tokens)
