import cv2
import numpy as np
import random
import os
from collections import deque
from ultralytics import YOLO
import hashlib

class VideoProcessor:
    def __init__(self, video_path, window_size=10, danger_img_dir="danger_frames"):
        self.video_path = video_path
        self.window_size = window_size
        self.danger_img_dir = danger_img_dir
        
        self._saved_hashes = set()
        self._saved_counts = {}
        # Cloud-safe directory creation
        try:
            os.makedirs(self.danger_img_dir, exist_ok=True)
            self._save_base = os.path.dirname(os.path.abspath(danger_img_dir)) or "."
        except OSError:
            # Fallback for cloud environments with strict permissions
            self.danger_img_dir = "/tmp/danger_frames"
            os.makedirs(self.danger_img_dir, exist_ok=True)
            self._save_base = "/tmp"

        try:
            self.model = YOLO("yolov8n.pt")
        except Exception as e:
            print(f"YOLO model load failed: {e}")
            self.model = None
        
        self.relevant_classes = [0, 1, 2, 3, 5, 7, 67]
        self.object_history = {}
        self.history_max_len = 5
        self.frame_count = 0
        self.detection_failures = 0
        self.total_frames_processed = 0
        
        self.handheld_phone_frames = 0
        self.mounted_phone_frames = 0

    def estimate_speed_heuristic(self, frame, prev_gray):
        """
        Calculates speed AND 'erratic_motion' (aggression).
        Returns: (speed_status, is_erratic)
        """
        if prev_gray is None:
            return "stationary", False
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Analyze center for forward speed
        h, w = gray.shape
        center_region = magnitude[h // 4:3 * h // 4, w // 3:2 * w // 3]
        avg_motion = np.mean(center_region) if center_region.size > 0 else np.mean(magnitude)
        
        # --- NEW: DETECT AGGRESSION (Flow Variance) ---
        # High variance = chaotic movement (swerving, shaking, hard braking)
        motion_variance = np.var(magnitude)
        is_erratic = motion_variance > 50.0  # Threshold for "rough ride"
        
        frame_width = frame.shape[1]
        scale = frame_width / 1280.0
        
        # --- UPDATED: TIGHTER THRESHOLDS ---
        # Lowered thresholds to catch "moderate" speed more easily
        if avg_motion < (1.5 * scale):
            return "stationary", False
        if avg_motion < (14.0 * scale): # tuned for Dhaka city speeds
            return "slow", is_erratic
        if avg_motion < (28.0 * scale):
            return "fast", is_erratic # Map mid-range to fast for safety
            
        return "fast", is_erratic

    def calculate_ttc(self, current_box, obj_id, frame_width, obj_class):
        curr_w = current_box[2] - current_box[0]
        cx = (current_box[0] + current_box[2]) / 2
        
        rough_pos = int(cx / frame_width * 10)
        obj_key = (obj_class, rough_pos)
        
        if obj_key not in self.object_history:
            self.object_history[obj_key] = deque(maxlen=self.history_max_len)
        
        history = self.object_history[obj_key]
        
        if len(history) > 0:
            prev_w = history[-1]
            if prev_w == 0 or curr_w == 0:
                history.append(curr_w)
                return "stable"
            
            expansion_rate = (curr_w - prev_w) / prev_w
            history.append(curr_w)
            
            if expansion_rate > 0.05:
                return "critical_approach"
            elif expansion_rate > 0.01:
                return "closing_in"
            else:
                return "stable"
        
        history.append(curr_w)
        return "unknown"

    def detect_glare(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        top_half = gray[0:h // 2, :]
        _, thresh = cv2.threshold(top_half, 240, 255, cv2.THRESH_BINARY)
        glare_pixels = np.sum(thresh == 255)
        glare_ratio = glare_pixels / top_half.size if top_half.size > 0 else 0
        return glare_ratio > 0.08

    def detect_darkness_improved(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        dark_pixels = np.sum(gray < 80)
        dark_ratio = dark_pixels / gray.size if gray.size > 0 else 0
        sky_region = gray[0:max(1, int(h * 0.1)), :]
        ground_region = gray[int(h * 0.5):h, :]
        sky_brightness = np.mean(sky_region) if sky_region.size > 0 else 128
        ground_brightness = np.mean(ground_region) if ground_region.size > 0 else 128
        is_night = (dark_ratio > 0.6 or (ground_brightness < 70 and (sky_brightness - ground_brightness) > 20))
        return is_night

    def detect_wet_or_glare_surface(self, frame, is_night):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        bottom_half = gray[h // 2:h, :]
        _, thresh = cv2.threshold(bottom_half, 200, 255, cv2.THRESH_BINARY)
        bright_ratio = np.sum(thresh == 255) / bottom_half.size if bottom_half.size > 0 else 0
        laplacian = cv2.Laplacian(bottom_half, cv2.CV_64F)
        texture_variance = np.var(laplacian)
        edges = cv2.Canny(bottom_half, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size if edges.size > 0 else 0
        
        wet_detected = False
        if is_night and bright_ratio > 0.05 and texture_variance < 500 and edge_density < 0.15:
            wet_detected = True
        if not is_night and bright_ratio > 0.12 and texture_variance < 300 and edge_density < 0.20:
            wet_detected = True
        return wet_detected

    def classify_phone_risk(self, boxes, classes, frame_width, frame_height):
        has_handheld = False
        has_mounted = False
        
        for box, cls in zip(boxes, classes):
            if cls != 67: continue
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            bh = y2 - y1
            bw = x2 - x1
            h_ratio = bh / frame_height
            y_center_ratio = cy / frame_height
            x_center_ratio = cx / frame_width
            aspect_ratio = bw / bh if bh > 0 else 1.0
            
            is_mounted = (y_center_ratio > 0.65 and h_ratio < 0.12 and 0.35 < x_center_ratio < 0.65 and 0.3 < aspect_ratio < 1.5)
            if is_mounted:
                has_mounted = True
                self.mounted_phone_frames += 1
                self.handheld_phone_frames = 0
            else:
                has_handheld = True
                self.handheld_phone_frames += 1
                self.mounted_phone_frames = 0
        
        if not has_handheld and not has_mounted:
            self.handheld_phone_frames = max(0, self.handheld_phone_frames - 1)
            self.mounted_phone_frames = max(0, self.mounted_phone_frames - 1)
        
        risk_level = self.get_phone_risk_level()
        return has_handheld, has_mounted, risk_level

    def get_phone_risk_level(self):
        # More sensitive thresholds: handheld use is highly risky
        if self.handheld_phone_frames >= 6:
            return "danger"
        elif self.handheld_phone_frames >= 4:
            return "caution"

        # Mounted phones are less risky but still notable if frequent
        if self.mounted_phone_frames >= 9:
            return "caution"
        elif self.mounted_phone_frames >= 8:
            return "caution"

        return "safe"

    def check_sandwich_condition(self, boxes, classes, frame_width):
        heavy_vehicles_x = []
        barriers_x = []
        for box, cls in zip(boxes, classes):
            cx = (box[0] + box[2]) / 2
            if cls in [5, 7]: heavy_vehicles_x.append(cx)
            elif cls == 2: barriers_x.append(cx)
        
        if heavy_vehicles_x and barriers_x:
            for hv_x in heavy_vehicles_x:
                for b_x in barriers_x:
                    if abs(hv_x - b_x) < (frame_width * 0.50): return True
        return False

    def _compute_dhaka_side_risks(self, boxes, classes, frame_width, frame_height):
        if not boxes: return False, False
        side_cut_risk = False
        wrong_side_risk = False
        center_lane_left = frame_width * 0.33
        center_lane_right = frame_width * 0.66

        for box, cls in zip(boxes, classes):
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            bw = box[2] - box[0]
            closeness = bw / (frame_width * 0.8)
            # side cut: object near extreme edges with notable width
            if closeness > 0.35 and (cx < frame_width * 0.15 or cx > frame_width * 0.85):
                side_cut_risk = True

            # wrong-side detection: require a larger closeness and that object is low in frame (closer)
            if cls in [2, 3, 5, 7] and center_lane_left < cx < center_lane_right:
                if closeness > 0.5 and cy > 0.6 * frame_height:
                    wrong_side_risk = True
        return side_cut_risk, wrong_side_risk

    def _compute_follow_distance_and_pinch(self, boxes, classes, frame_width, frame_height):
        short_follow_distance = False
        pinch_point = False
        bus_blind_spot = False
        unsecured_load_risk = False
        if not boxes: return short_follow_distance, pinch_point, bus_blind_spot, unsecured_load_risk
        
        center_lane_left = frame_width * 0.30
        center_lane_right = frame_width * 0.70
        closest_front_width = 0
        closest_front_is_truck = False
        left_close = False
        right_close = False
        
        for box, cls in zip(boxes, classes):
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            bw = box[2] - box[0]
            width_ratio = bw / frame_width
            is_vehicle = cls in [2, 3, 5, 7]
            is_heavy = cls in [5, 7]
            
            if is_vehicle and center_lane_left < cx < center_lane_right:
                if width_ratio > closest_front_width:
                    closest_front_width = width_ratio
                    closest_front_is_truck = (cls == 7)
            if is_vehicle and width_ratio > 0.25:
                if cx < frame_width * 0.35: left_close = True
                if cx > frame_width * 0.65: right_close = True
            if is_heavy and 0.35 * frame_height < cy < 0.75 * frame_height and width_ratio > 0.25:
                if 0.2 < cx / frame_width < 0.8: bus_blind_spot = True
        
        if closest_front_width > (0.50 * (frame_width / 1280.0)): short_follow_distance = True
        if left_close and right_close: pinch_point = True
        if closest_front_is_truck and 0.25 < closest_front_width <= 0.45: unsecured_load_risk = True
        return short_follow_distance, pinch_point, bus_blind_spot, unsecured_load_risk

    def _compute_entering_traffic_and_pedestrian(self, boxes, classes, frame_width, frame_height, ttc_status):
        entering_traffic_risk = False
        pedestrian_crossing_risk = False
        if not boxes: return entering_traffic_risk, pedestrian_crossing_risk
        
        for box, cls in zip(boxes, classes):
            cx = (box[0] + box[2]) / 2
            bw = box[2] - box[0]
            bh = box[3] - box[1]
            h_ratio = bh / frame_height
            y_bottom_ratio = box[3] / frame_height
            x_center_ratio = cx / frame_width
            width_ratio = bw / frame_width
            
            if cls == 0:
                close_and_central = (h_ratio > 0.25 and y_bottom_ratio > 0.7 and 0.15 < x_center_ratio < 0.85)
                if close_and_central or (ttc_status == "critical_approach"): pedestrian_crossing_risk = True
            
            if cls in [2, 3] and width_ratio > 0.20:
                extreme_edge = (cx < frame_width * 0.15 or cx > frame_width * 0.85)
                diagonal_from_center = (0.3 < x_center_ratio < 0.7 and y_bottom_ratio < 0.5)
                if (extreme_edge or diagonal_from_center) and ttc_status == "critical_approach": entering_traffic_risk = True
        return entering_traffic_risk, pedestrian_crossing_risk

    def save_danger_frame(self, frame, frame_id, reason="danger", subdir=None, max_images=None):
        target_dir = subdir if subdir is not None else self.danger_img_dir
        if not os.path.isabs(target_dir): target_dir = os.path.join(self._save_base, target_dir)
        try: os.makedirs(target_dir, exist_ok=True)
        except: return None
        ok, enc = cv2.imencode('.jpg', frame)
        if not ok: return None
        img_bytes = enc.tobytes()
        h = hashlib.md5(img_bytes).hexdigest()
        cnt = self._saved_counts.get(target_dir, 0)
        if h in self._saved_hashes: return None
        if max_images is not None and cnt >= max_images: return None
        filename = f"{reason}_frame_{frame_id:06d}_{cnt}.jpg"
        path = os.path.join(target_dir, filename)
        try:
            with open(path, "wb") as f: f.write(img_bytes)
            self._saved_hashes.add(h)
            self._saved_counts[target_dir] = cnt + 1
            return path
        except: return None

    def save_frame_by_risk(self, frame, frame_id, risk_label):
        label_map = {0: "safe_frames", 1: "caution_frames", 2: "danger_frames", "safe": "safe_frames", "caution": "caution_frames", "danger": "danger_frames"}
        folder = label_map.get(risk_label, "danger_frames")
        return self.save_danger_frame(frame, frame_id, reason=folder.rstrip("_frames"), subdir=folder)

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        frame_data = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Determine sampling strategy
        if fps is None or fps <= 0 or total_frames <= 0:
            # Fallback for bad metadata
            if self.window_size <= 0: self.window_size = 30
            num_windows = max(total_frames // self.window_size, 1)
            for i in range(num_windows):
                start = i * self.window_size
                end = min((i + 1) * self.window_size, total_frames)
                if end - start < 2: continue
                idx = random.randint(start, end - 2)
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret1, frame1 = cap.read()
                ret2, frame2 = cap.read()
                if not ret1 or not ret2: break
                self.total_frames_processed += 1
                prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                # Pass BOTH frames to heuristic to get speed + variance
                speed_status, is_erratic = self.estimate_speed_heuristic(frame2, prev_gray)
                frame_data.append(self._analyze_single_frame(frame2, idx, speed_status, is_erratic))
                del frame1, frame2
        else:
            # Standard 2 frames per second
            duration = max(int(np.ceil(total_frames / max(1.0, fps))), 1)
            print(f"Sampling 2 frames/sec for ~{duration} seconds...")
            for s in range(duration):
                idx1 = int(s * fps)
                idx2 = int(min((s + 1) * fps - 1, total_frames - 1))
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx1)
                ret1, frame1 = cap.read()
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx2)
                ret2, frame2 = cap.read()
                
                if not ret1 or not ret2: continue
                self.total_frames_processed += 2
                
                prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                speed_status, is_erratic = self.estimate_speed_heuristic(frame2, prev_gray)
                
                frame_data.append(self._analyze_single_frame(frame1, idx1, speed_status, is_erratic))
                if idx2 != idx1:
                    frame_data.append(self._analyze_single_frame(frame2, idx2, speed_status, is_erratic))
                del frame1, frame2
        
        cap.release()
        return frame_data

    def _analyze_single_frame(self, frame, frame_idx, speed_status, is_erratic):
        # ... Object detection logic ...
        objects_detected = []
        max_closeness = 0.0
        ttc_status = "stable"
        current_boxes = []
        current_classes = []
        results = None
        
        if self.model:
            try: results = self.model(frame, verbose=False)[0]
            except: self.detection_failures += 1
        
        if results:
            for idx, box in enumerate(results.boxes):
                cls = int(box.cls[0])
                if cls in self.relevant_classes:
                    label = self.model.names.get(cls, str(cls))
                    if label == "bicycle": label = "rickshaw"
                    x1,y1,x2,y2 = map(float, box.xyxy[0])
                    current_boxes.append([x1,y1,x2,y2])
                    current_classes.append(cls)
                    closeness = min((x2-x1) / (frame.shape[1] * 0.8), 1.0)
                    max_closeness = max(max_closeness, closeness)
                    objects_detected.append(label)
                    if self.calculate_ttc([x1,y1,x2,y2], idx, frame.shape[1], cls) == "critical_approach":
                        ttc_status = "critical_approach"

        has_handheld, has_mounted, phone_risk = self.classify_phone_risk(current_boxes, current_classes, frame.shape[1], frame.shape[0])
        is_glare = self.detect_glare(frame)
        is_night = self.detect_darkness_improved(frame)
        is_sandwich = self.check_sandwich_condition(current_boxes, current_classes, frame.shape[1])
        side_cut, wrong_side = self._compute_dhaka_side_risks(current_boxes, current_classes, frame.shape[1], frame.shape[0])
        short_follow, pinch, bus_blind, unsecured = self._compute_follow_distance_and_pinch(current_boxes, current_classes, frame.shape[1], frame.shape[0])
        entering, ped_cross = self._compute_entering_traffic_and_pedestrian(current_boxes, current_classes, frame.shape[1], frame.shape[0], ttc_status)
        wet_glare = self.detect_wet_or_glare_surface(frame, is_night)
        
        late_night_high_speed = (is_night and speed_status == "fast" and len(objects_detected) <= 1)

        return {
            "frame_id": frame_idx,
            "frame": frame.copy(),
            "objects": objects_detected,
            "proximity_score": max_closeness,
            "ego_speed": speed_status,
            "is_erratic": is_erratic,  # <--- NEW FIELD
            "ttc_status": ttc_status,
            "glare": is_glare,
            "night": is_night,
            "phone_detected": has_handheld,
            "phone_mounted": has_mounted,
            "phone_risk": phone_risk,
            "sandwich_risk": is_sandwich,
            "side_cut_risk": side_cut,
            "wrong_side_risk": wrong_side,
            "short_follow_distance": short_follow,
            "pinch_point": pinch,
            "bus_blind_spot": bus_blind,
            "unsecured_load_risk": unsecured,
            "entering_traffic_risk": entering,
            "pedestrian_crossing_risk": ped_cross,
            "wet_or_glare_surface": wet_glare,
            "late_night_high_speed": late_night_high_speed,
        }