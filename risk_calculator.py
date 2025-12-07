class RiskCalculator:
    def __init__(self):
        self.STATIONARY = 0
        self.SLOW = 15 
        self.MODERATE = 40
        self.FAST = 60
    
    def calculate_risk_score(self, frame_data, ego_speed_category="slow"):
        risk_score = 0
        
        # --- 0. NEW: AGGRESSIVE/ERRATIC MANEUVER ---
        # If camera is shaking/swerving wildly, immediate high risk
        # This overrides "safe slow speed" logic.
        is_erratic = frame_data.get("is_erratic", False)
        if is_erratic:
            risk_score += 35 

        # --- 1. BASE ENVIRONMENTAL RISKS ---
        # Reduce baseline environmental contributions so they don't push low-speed frames over the edge
        if frame_data.get("glare", False): risk_score += 8
        if frame_data.get("night", False): risk_score += 6
        if frame_data.get("wet_or_glare_surface", False): risk_score += 8
        
        # --- 2. PHONE DISTRACTION ---
        # Trust temporal phone risk more than a single-frame detector
        phone_risk = frame_data.get("phone_risk", None)
        if phone_risk == "danger":
            risk_score += 70
        elif phone_risk == "caution":
            risk_score += 30
        elif frame_data.get("phone_detected", False):
            risk_score += 30
        
        # --- 3. PROXIMITY RISKS ---
        proximity = frame_data.get("proximity_score", 0.0)
        ttc_status = frame_data.get("ttc_status", "stable")
        
        if ego_speed_category == "stationary":
            if proximity > 0.6: risk_score += 5
        
        elif ego_speed_category == "slow":
            # More conservative additions for slow/traffic jam scenarios
            if proximity > 0.5: risk_score += 2
            elif proximity > 0.4: risk_score += 1
            if ttc_status == "critical_approach":
                risk_score += 12
            elif ttc_status == "closing_in":
                risk_score += 4
        
        elif ego_speed_category == "moderate": # Adjusted for Dhaka
            if proximity > 0.5: risk_score += 18
            elif proximity > 0.4: risk_score += 10
            elif proximity > 0.2: risk_score += 5
            if ttc_status == "critical_approach":
                risk_score += 20
            elif ttc_status == "closing_in":
                risk_score += 8
        
        elif ego_speed_category == "fast":
            if proximity > 0.5: risk_score += 35
            elif proximity > 0.4: risk_score += 25
            elif proximity > 0.2: risk_score += 15
            if ttc_status == "critical_approach": risk_score += 40
            elif ttc_status == "closing_in": risk_score += 25
        
        # --- 4. OBJECT-BASED RISKS ---
        objects = frame_data.get("objects", [])
        has_heavy = any(o in ["bus", "truck", "heavy_vehicle"] for o in objects)
        has_rickshaw = "rickshaw" in objects
        
        if has_heavy:
            if ego_speed_category == "fast": risk_score += 15
            elif ego_speed_category == "moderate": risk_score += 8
            else: risk_score += 3
        if has_rickshaw: risk_score += 5
        
        # --- 5. DHAKA HAZARDS ---
        if frame_data.get("wrong_side_risk", False):
            risk_score += 35 if ego_speed_category == "fast" else 15
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
        
        # --- 6. PEDESTRIAN ---
        # Tuned for Dhaka: jam + pedestrians are common and should not always be near-100
        ped_risk = frame_data.get("pedestrian_crossing_risk", False)
        if ped_risk:
            if ego_speed_category in ("fast", "moderate"):
                risk_score += 35
            else:
                # jam / slow typical Dhaka
                base = 8
                if ttc_status == "critical_approach":
                    base += 15
                elif ttc_status == "closing_in":
                    base += 5
                risk_score += base

        # --- 7. LATE NIGHT ---
        if frame_data.get("late_night_high_speed", False): risk_score += 25
        
        # --- 8. TRAFFIC JAM EXCEPTION (STRONGER) ---
        # Reduce risk more aggressively for true jams when not erratic
        if (ego_speed_category in ("stationary", "slow")) and not is_erratic:
            if proximity > 0.4 and ttc_status != "critical_approach":
                reduction = min(40, int(risk_score * 0.50))
                risk_score = max(0, risk_score - reduction)

        # Guard: if score is very high but no critical binary hazard flags, cap it
        has_critical_flag = (
            frame_data.get("wrong_side_risk", False)
            or frame_data.get("side_cut_risk", False)
            or frame_data.get("sandwich_risk", False)
            or frame_data.get("entering_traffic_risk", False)
            or (frame_data.get("pedestrian_crossing_risk", False) and ego_speed_category in ("moderate", "fast"))
            or (frame_data.get("phone_risk", "") == "danger")
        )
        if not has_critical_flag and risk_score > 70:
            risk_score = 70

        risk_score = min(100, max(0, int(risk_score)))
        return risk_score
    
    def score_to_label(self, risk_score):
        # More conservative thresholds to avoid over-optimistic SAFE labels
        if risk_score < 25: return "SAFE"
        elif risk_score < 55: return "CAUTION"
        else: return "DANGER"

    def get_risk_details(self, risk_score):
        label = self.score_to_label(risk_score)
        if label == "SAFE": detail = "Normal riding conditions"
        elif label == "CAUTION": detail = "Minor risk factors present"
        else: detail = "Significant risk detected"
        return {"score": risk_score, "label": label, "detail": detail}