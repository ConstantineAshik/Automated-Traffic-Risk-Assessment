class RiskCalculator:
    def __init__(self):
        self.STATIONARY = 0
        self.SLOW = 15
        self.MODERATE = 40
        self.FAST = 60

    def calculate_risk_score(self, frame_data, ego_speed_category="slow"):
        risk_score = 0

        is_erratic = frame_data.get("is_erratic", False)
        if is_erratic:
            risk_score += 35

        if frame_data.get("glare", False):
            risk_score += 8
        if frame_data.get("night", False):
            risk_score += 6
        if frame_data.get("wet_or_glare_surface", False):
            risk_score += 8

        phone_risk = frame_data.get("phone_risk", None)
        if phone_risk == "danger":
            risk_score += 70
        elif phone_risk == "caution":
            risk_score += 30
        elif frame_data.get("phone_detected", False):
            risk_score += 30

        proximity = frame_data.get("proximity_score", 0.0)
        side_proximity = frame_data.get("side_proximity_score", 0.0)
        traffic_jam = frame_data.get("traffic_jam", False)
        front_ttc = frame_data.get("front_ttc_seconds")
        front_relative_speed = frame_data.get("front_relative_speed_proxy", 0.0)
        front_stable_seconds = frame_data.get("front_stable_seconds", 0.0)
        ttc_status = frame_data.get("ttc_status", "stable")

        # Motion context matters more than one-frame distance. A close vehicle
        # in front during stopped traffic should not become DANGER unless the
        # rider is actually closing in.
        if proximity >= 0.55:
            risk_score += 8 if ego_speed_category in ("stationary", "slow") else 16
        elif proximity > 0.35:
            risk_score += 3 if ego_speed_category in ("stationary", "slow") else 8
        elif proximity > 0.20 and ego_speed_category == "fast":
            risk_score += 5

        if side_proximity > 0.45 and not proximity > 0.25:
            risk_score += 3

        if front_ttc is not None:
            if front_ttc < 1.5:
                risk_score += 45
            elif front_ttc <= 3.0:
                risk_score += 25
            elif front_ttc <= 5.0 and ego_speed_category == "fast":
                risk_score += 8
        elif ttc_status == "critical_approach":
            risk_score += 30
        elif ttc_status == "closing_in":
            risk_score += 15

        if front_relative_speed > 0.12:
            risk_score += 22
        elif front_relative_speed > 0.06:
            risk_score += 12

        if traffic_jam:
            risk_score -= 30
        if ego_speed_category in ("stationary", "slow"):
            risk_score -= 10
        if front_stable_seconds >= 3.0:
            risk_score -= 20
        elif front_stable_seconds >= 1.5:
            risk_score -= 10

        objects = frame_data.get("objects", [])
        has_heavy = any(o in ("bus", "truck", "heavy_vehicle") for o in objects)
        has_rickshaw = any(
            o in ("rickshaw", "auto rickshaw", "cng") for o in objects
        )

        if has_heavy:
            if ego_speed_category == "fast":
                risk_score += 15
            elif ego_speed_category == "moderate":
                risk_score += 8
            else:
                risk_score += 3
        if has_rickshaw:
            risk_score += 5

        if frame_data.get("wrong_side_risk", False):
            risk_score += 35 if ego_speed_category == "fast" else 15
        if frame_data.get("frontal_conflict_risk", False):
            risk_score += 30 if ego_speed_category == "fast" else 15
        if frame_data.get("side_cut_risk", False):
            risk_score += 28 if ego_speed_category == "fast" else 10
        if frame_data.get("sandwich_risk", False):
            risk_score += 20 if ego_speed_category != "slow" else 5
        if frame_data.get("pinch_point", False):
            risk_score += 25 if ego_speed_category == "fast" else 10
        if frame_data.get("bus_blind_spot", False):
            risk_score += 22 if ego_speed_category == "fast" else 12
        if frame_data.get("unsecured_load_risk", False):
            risk_score += 18
        if frame_data.get("entering_traffic_risk", False):
            risk_score += 30 if ego_speed_category == "fast" else 15

        ped_risk = frame_data.get("pedestrian_crossing_risk", False)
        if ped_risk:
            if ego_speed_category in ("fast", "moderate"):
                risk_score += 35
            else:
                base = 8
                if ttc_status == "critical_approach":
                    base += 15
                elif ttc_status == "closing_in":
                    base += 5
                risk_score += base

        if frame_data.get("late_night_high_speed", False):
            risk_score += 25

        if ego_speed_category in ("stationary", "slow") and not is_erratic:
            if proximity > 0.4 and ttc_status != "critical_approach":
                reduction = min(40, int(risk_score * 0.50))
                risk_score = max(0, risk_score - reduction)

        has_critical_flag = (
            frame_data.get("wrong_side_risk", False)
            or frame_data.get("frontal_conflict_risk", False)
            or frame_data.get("side_cut_risk", False)
            or frame_data.get("sandwich_risk", False)
            or frame_data.get("entering_traffic_risk", False)
            or (
                frame_data.get("pedestrian_crossing_risk", False)
                and ego_speed_category in ("moderate", "fast")
            )
            or (front_ttc is not None and front_ttc < 1.5)
            or front_relative_speed > 0.12
            or (frame_data.get("phone_risk", "") == "danger")
        )
        if not has_critical_flag and risk_score > 70:
            risk_score = 70

        risk_score = min(100, max(0, int(risk_score)))
        return risk_score

    def score_to_label(self, risk_score):
        if risk_score < 25:
            return "SAFE"
        if risk_score < 55:
            return "CAUTION"
        return "DANGER"

    def get_risk_details(self, risk_score):
        label = self.score_to_label(risk_score)
        if label == "SAFE":
            detail = "Normal riding conditions"
        elif label == "CAUTION":
            detail = "Minor risk factors present"
        else:
            detail = "Significant risk detected"
        return {"score": risk_score, "label": label, "detail": detail}
