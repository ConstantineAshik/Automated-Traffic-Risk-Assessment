from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

class RiskModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=100, min_df=1, max_df=0.9)
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'  # FIX #20: Handle class imbalance
        )
        self.is_trained = False
    
    def train_mock_model(self):
        """
        Trains model based on context-aware patterns.
        
        Labels:
        - 0: SAFE
        - 1: CAUTION
        - 2: DANGER
        
        FIX #20: Better balanced training data.
        FIX #21: Fixed label array mismatch issue.
        """
        descriptions = [
            # --- SAFE SCENARIOS (0) ---
            "day safe_gap fast",
            "day traffic_jam_proximity heavy_vehicle slow",
            "night traffic_jam_proximity rickshaw stationary",
            "day low_speed_filtering sandwich_risk",
            "day safe_gap slow",
            "day safe_gap moderate_distance slow",
            # Additional Dhaka-safe examples: jam + pedestrians at slow speed
            "day traffic_jam_proximity pedestrian_crossing slow",
            "day traffic_jam_proximity pedestrian_crossing stationary",
            "day traffic_jam_proximity pedestrian_crossing safe_gap slow",
            "night safe_gap stationary",
            "day traffic_jam_proximity moderate_distance slow",
            
            # --- MODERATE RISKS (1) ---
            "day moderate_distance heavy_vehicle slow",
            "day moderate_distance heavy_vehicle moderate_speed",
            "night low_speed_filtering rickshaw",
            "night wet_road_glare safe_gap",
            "night late_night_high_speed_open_road safe_gap",
            "night pedestrian_crossing moderate_distance",
            "day side_cut_risk moderate_distance",
            "day wrong_side_risk safe_gap",
            "day short_follow_distance slow",
            "day pinch_point_no_escape slow moderate_distance",
            "day short_follow_distance moderate_distance slow",
            "night bus_blind_spot moderate_distance",
            
            # --- CRITICAL RISKS (2) ---
            "day high_speed_tailgating heavy_vehicle",
            "day high_speed_sandwich_risk",
            "day rapid_closing_speed safe_gap",
            "night phone_distraction",
            "day glare_blindness moderate_speed",
            "day high_speed_tailgating rickshaw",
            "night short_follow_distance high_speed_tailgating",
            "day short_follow_distance heavy_vehicle",
            "day pinch_point_no_escape heavy_vehicle",
            "night bus_blind_spot heavy_vehicle",
            "night unsecured_truck_load moderate_distance",
            "night entering_traffic_conflict rapid_closing_speed",
            "night pedestrian_crossing high_speed_tailgating",
            "night wet_road_glare high_speed_tailgating",
            "night late_night_high_speed_open_road high_speed_tailgating",
            "day side_cut_risk high_speed_tailgating",
            "day wrong_side_risk rapid_closing_speed",
            "night bus_blind_spot rapid_closing_speed",
            "day phone_distraction",  # Phone is always dangerous
            "day glare_blindness high_speed_tailgating",
        ]
        
        # FIX #21: Create balanced labels matching description count
        total = len(descriptions)
        
        # Target distribution: 35% SAFE, 35% CAUTION, 30% DANGER
        safe_count = int(total * 0.35)
        caution_count = int(total * 0.35)
        danger_count = total - safe_count - caution_count
        
        labels = [0] * safe_count + [1] * caution_count + [2] * danger_count
        
        # FIX #20: Verify alignment
        assert len(descriptions) == len(labels), f"Mismatch: {len(descriptions)} descriptions, {len(labels)} labels"
        
        print(f"Training on {total} samples: {safe_count} SAFE, {caution_count} CAUTION, {danger_count} DANGER")
        
        X_train = self.vectorizer.fit_transform(descriptions)
        self.classifier.fit(X_train, labels)
        
        self.is_trained = True
        print("✓ Model trained with Context-Aware Dhaka rules")
    
    def predict_risk(self, text_descriptions):
        """Predict risk levels for text descriptions."""
        if not self.is_trained:
            raise Exception("Model not trained. Call train_mock_model() first.")
        
        X_new = self.vectorizer.transform(text_descriptions)
        return self.classifier.predict(X_new)
    
    def predict_risk_proba(self, text_descriptions):
        """Get probability scores for each risk class."""
        if not self.is_trained:
            raise Exception("Model not trained. Call train_mock_model() first.")
        
        X_new = self.vectorizer.transform(text_descriptions)
        
        if hasattr(self.classifier, "predict_proba"):
            return self.classifier.predict_proba(X_new)
        return None
    
    def interpret_risk(self, risk_level):
        """Convert numeric risk level to human-readable label."""
        mapping = {0: "SAFE", 1: "CAUTION", 2: "DANGER"}
        return mapping.get(risk_level, "UNKNOWN")


