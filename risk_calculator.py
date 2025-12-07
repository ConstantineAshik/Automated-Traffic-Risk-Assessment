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
        if frame_data.get("glare", False): risk_score += 15
        if frame_data.get("night", False): risk_score += 10
        if frame_data.get("wet_or_glare_surface", False): risk_score += 12
        
        # --- 2. PHONE DISTRACTION ---
        if frame_data.get("phone_detected", False): risk_score += 40
        
        # --- 3. PROXIMITY RISKS ---
        proximity = frame_data.get("proximity_score", 0.0)
        ttc_status = frame_data.get("ttc_status", "stable")
        
        if ego_speed_category == "stationary":
            if proximity > 0.6: risk_score += 5
        
        elif ego_speed_category == "slow":
            if proximity > 0.5: risk_score += 3
            elif proximity > 0.4: risk_score += 1
            if ttc_status == "critical_approach": risk_score += 8
            elif ttc_status == "closing_in": risk_score += 3
        
        elif ego_speed_category == "moderate": # Adjusted up for rough rides
            if proximity > 0.5: risk_score += 25 # Increased
            elif proximity > 0.4: risk_score += 15 # Increased
            elif proximity > 0.2: risk_score += 8
            if ttc_status == "critical_approach": risk_score += 25
            elif ttc_status == "closing_in": risk_score += 12
        
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
        if frame_data.get("pedestrian_crossing_risk", False):
            risk_score += 35 if ego_speed_category == "fast" else 15

        # --- 7. LATE NIGHT ---
        if frame_data.get("late_night_high_speed", False): risk_score += 25
        
        # --- 8. TRAFFIC JAM EXCEPTION (MODIFIED) ---
        # ONLY reduce score if driving is NOT erratic
        if (ego_speed_category == "stationary" or ego_speed_category == "slow") and not is_erratic:
            if proximity > 0.4 and ttc_status != "critical_approach":
                reduction = min(20, int(risk_score * 0.30))
                risk_score = max(0, risk_score - reduction)
        
        risk_score = min(100, max(0, int(risk_score)))
        return risk_score
    
    def score_to_label(self, risk_score):
        if risk_score < 30: return "SAFE"
        elif risk_score < 65: return "CAUTION"
        else: return "DANGER"

    def get_risk_details(self, risk_score):
        label = self.score_to_label(risk_score)
        if label == "SAFE": detail = "Normal riding conditions"
        elif label == "CAUTION": detail = "Minor risk factors present"
        else: detail = "Significant risk detected"
        return {"score": risk_score, "label": label, "detail": detail}