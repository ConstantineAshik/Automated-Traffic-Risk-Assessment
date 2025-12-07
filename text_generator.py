class TextGenerator:
    def __init__(self):
        pass

    def estimate_speed_category(self, speed_status):
        """Convert ego_speed to category for better context."""
        if speed_status == "stationary":
            return "stationary"
        elif speed_status == "slow":
            return "slow"
        elif speed_status == "fast":
            return "fast"
        else:
            return "slow"

    def generate_description(self, frame_data):
        """
        Generates context-aware description tokens for risk classification.
        
        Key insight: proximity alone ≠ danger in Dhaka traffic jams.
        Danger = (high speed OR rapid closing) + (obstacle OR pedestrian OR head-on).
        
        NEW: Phone risk now uses temporal classification (safe/caution/danger)
        """
        tokens = []

        speed = frame_data.get("ego_speed", "stationary")
        proximity = frame_data.get("proximity_score", 0.0)
        ttc_status = frame_data.get("ttc_status", "stable")
        objs = frame_data.get("objects", [])
        phone_risk = frame_data.get("phone_risk", "safe")  # NEW: temporal phone risk

        # Precompute flags
        has_heavy_obj = any(o in ("bus", "truck", "heavy_vehicle") for o in objs)
        has_pedestrian = "person" in objs
        has_rickshaw = "rickshaw" in objs

        # --- ENVIRONMENT ---
        if frame_data.get("glare", False):
            tokens.append("glare_blindness")
        elif frame_data.get("night", False):
            tokens.append("night")
            if frame_data.get("wet_or_glare_surface", False):
                tokens.append("wet_road_glare")
        else:
            tokens.append("day")

        # --- PHONE DETECTION (IMPROVED with temporal tracking) ---
        # Only flag as "phone_distraction" if temporal risk is danger
        if phone_risk == "danger":
            tokens.append("phone_distraction")
        elif phone_risk == "caution":
            tokens.append("phone_usage_caution")  # brief check, not immediate danger

        # --- SPEED CONTEXT ---
        tokens.append(speed)

        # --- TRAFFIC JAM DETECTION (Dhaka-specific) ---
        # If slow/stationary + moderate-high proximity = likely traffic jam (SAFE)
        is_traffic_jam = speed in ("slow", "stationary") and proximity > 0.3

        if is_traffic_jam:
            tokens.append("traffic_jam_proximity")
            # Add consistent tokens used by the risk model
            if has_pedestrian:
                tokens.append("pedestrian_crossing")
            if frame_data.get("short_follow_distance", False):
                tokens.append("short_follow_distance")
            if frame_data.get("bus_blind_spot", False):
                tokens.append("bus_blind_spot")
            # Use 'safe_gap' as the canonical safe-distance token
            if proximity <= 0.3:
                tokens.append("safe_gap")
            elif proximity <= 0.5:
                tokens.append("moderate_distance")
            else:
                tokens.append("close_proximity")
            return " ".join(tokens)  # Exit early: traffic jam context overrides proximity risk

        # --- HIGH SPEED SCENARIOS (actual danger) ---
        # Only add explicit high-risk tokens if multiple signals support it
        if speed == "fast":
            # require at least one additional indicator to tag as high_speed
            high_speed_support = (
                frame_data.get("is_erratic", False)
                or ttc_status == "critical_approach"
                or ttc_status == "closing_in"
                or proximity > 0.45
            )
            if high_speed_support:
                tokens.append("high_speed")

            # Head-on or side risk (vehicle in wrong lane/extreme position)
            if frame_data.get("wrong_side_risk", False):
                tokens.append("wrong_side_risk")
            if frame_data.get("side_cut_risk", False):
                tokens.append("side_cut_risk")

            # Rapid approach to obstacle - only when TTC indicates real closing
            if ttc_status == "critical_approach":
                tokens.append("rapid_closing_speed")

            # High speed + pedestrian = DANGER only if closing or erratic
            if has_pedestrian and (ttc_status == "critical_approach" or frame_data.get("is_erratic", False)):
                tokens.append("pedestrian_crossing")

            # High speed + sandwich/pinch = DANGER
            if frame_data.get("sandwich_risk", False):
                tokens.append("high_speed_sandwich_risk")
            if frame_data.get("pinch_point", False):
                tokens.append("pinch_point")

            # High speed + heavy vehicle tailgating = DANGER
            if proximity > 0.4 and has_heavy_obj:
                tokens.append("high_speed_tailgating")

            # High speed at night with low visibility
            if frame_data.get("night", False) and len(objs) <= 1 and high_speed_support:
                tokens.append("late_night_high_speed")

        # --- MODERATE SPEED (slow/moderate) ---
        else:
            # Pedestrian crossing even at slow speed = risky
            if has_pedestrian and proximity > 0.25:
                tokens.append("pedestrian_crossing")

            # short follow distance and pinch tokens for model alignment
            if frame_data.get("short_follow_distance", False):
                tokens.append("short_follow_distance")
            if frame_data.get("pinch_point", False):
                tokens.append("pinch_point_no_escape")

            # Bus blind spot at any speed if close
            if frame_data.get("bus_blind_spot", False) and proximity > 0.3:
                tokens.append("bus_blind_spot")

            # Entering traffic conflict (diagonal vehicle)
            if frame_data.get("entering_traffic_risk", False):
                tokens.append("entering_traffic_conflict")

        # --- GENERIC MODERATE-RISK TOKENS ---
        if proximity > 0.5:
            tokens.append("close_proximity")
        elif proximity > 0.3:
            tokens.append("moderate_distance")
        else:
            # Use canonical safe token 'safe_gap' to match training
            tokens.append("safe_gap")

        return " ".join(tokens)