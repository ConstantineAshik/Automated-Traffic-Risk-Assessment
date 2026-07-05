import cv2
import numpy as np
import os
import hashlib
from pathlib import Path

from vision.ensemble import Detection, EnsembleDetector
from vision.tracking import IoUTracker

class VideoProcessor:
    def __init__(
        self,
        video_path,
        window_size=10,
        danger_img_dir="danger_frames",
        sampling_fps=2.0,
        max_frames=0,
        model_paths=("models/yolo11n.pt", "models/yolo11m.pt", "models/yolo12m.pt"),
        detection_confidence=0.25,
        ensemble_iou_threshold=0.55,
        ensemble_min_model_votes=1,
        inference_image_size=640,
        inference_device=None,
    ):
        self.video_path = video_path
        self.window_size = window_size
        self.danger_img_dir = danger_img_dir
        self.sampling_fps = sampling_fps
        self.max_frames = max_frames
        
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

        project_root = Path(__file__).resolve().parent
        resolved_model_paths = [
            str(
                (project_root / path).resolve()
                if not Path(path).is_absolute()
                else Path(path)
            )
            for path in model_paths
        ]
        self.detector = EnsembleDetector(
            model_paths=resolved_model_paths,
            confidence=detection_confidence,
            iou_threshold=ensemble_iou_threshold,
            min_model_votes=ensemble_min_model_votes,
            image_size=inference_image_size,
            device=inference_device,
        )
        self.detector_metadata = self.detector.metadata()
        self.tracker = IoUTracker(
            minimum_iou=0.10,
            max_age_seconds=max(2.0, 3.0 / max(self.sampling_fps, 0.1)),
        )
        self.frame_count = 0
        self.detection_failures = 0
        self.total_frames_processed = 0
        self.inference_errors = {}
        
        self.handheld_phone_frames = 0
        self.mounted_phone_frames = 0

    def estimate_speed_heuristic(self, frame, prev_gray, elapsed_seconds):
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
        
        frame_width = frame.shape[1]
        scale = max(frame_width / 1280.0, 0.1)
        elapsed_seconds = max(float(elapsed_seconds), 1 / 120)
        motion_per_second = avg_motion / elapsed_seconds
        variance_per_second = np.var(magnitude) / (elapsed_seconds**2)
        is_erratic = variance_per_second > (200.0 * scale**2)

        if motion_per_second < (3.0 * scale):
            return "stationary", False
        if motion_per_second < (28.0 * scale):
            return "slow", is_erratic
        return "fast", is_erratic

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

    def classify_phone_risk(self, detections, frame_width, frame_height):
        has_handheld = False
        has_mounted = False
        people = [d.box for d in detections if d.label == "person"]

        for detection in detections:
            if detection.label != "cell phone":
                continue
            box = detection.box
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            bh = y2 - y1
            bw = x2 - x1
            h_ratio = bh / frame_height
            y_center_ratio = cy / frame_height
            x_center_ratio = cx / frame_width
            aspect_ratio = bw / bh if bh > 0 else 1.0
            
            overlaps_person = any(
                person[0] - bw <= cx <= person[2] + bw
                and person[1] - bh <= cy <= person[3] + bh
                for person in people
            )
            is_mounted = (
                y_center_ratio > 0.65
                and h_ratio < 0.12
                and 0.35 < x_center_ratio < 0.65
                and 0.3 < aspect_ratio < 1.5
                and not overlaps_person
            )
            if is_mounted:
                has_mounted = True
            elif overlaps_person:
                has_handheld = True

        if has_handheld:
            self.handheld_phone_frames += 1
            self.mounted_phone_frames = 0
        elif has_mounted:
            self.mounted_phone_frames += 1
            self.handheld_phone_frames = 0
        else:
            self.handheld_phone_frames = 0
            self.mounted_phone_frames = 0
        
        risk_level = self.get_phone_risk_level()
        return has_handheld, has_mounted, risk_level

    def get_phone_risk_level(self):
        danger_frames = max(2, int(round(3.0 * self.sampling_fps)))
        caution_frames = max(2, int(round(1.5 * self.sampling_fps)))
        mounted_caution_frames = max(3, int(round(4.0 * self.sampling_fps)))

        if self.handheld_phone_frames >= danger_frames:
            return "danger"
        if self.handheld_phone_frames >= caution_frames:
            return "caution"
        if self.mounted_phone_frames >= mounted_caution_frames:
            return "caution"

        return "safe"

    def check_sandwich_condition(self, detections, frame_width):
        left_heavy = False
        right_heavy = False
        for detection in detections:
            if detection.label not in ("bus", "truck"):
                continue
            x1, _, x2, _ = detection.box
            width_ratio = (x2 - x1) / frame_width
            center_ratio = ((x1 + x2) / 2) / frame_width
            if width_ratio < 0.18:
                continue
            left_heavy |= center_ratio < 0.45
            right_heavy |= center_ratio > 0.55
        return left_heavy and right_heavy

    def _compute_dhaka_side_risks(
        self, detections, frame_width, frame_height
    ):
        if not detections:
            return False, False, False
        side_cut_risk = False
        wrong_side_risk = False
        frontal_conflict_risk = False
        center_lane_left = frame_width * 0.33
        center_lane_right = frame_width * 0.66

        for detection in detections:
            if detection.label not in (
                "car",
                "motorcycle",
                "bus",
                "truck",
                "rickshaw",
                "wrong way vehicle",
            ):
                continue
            box = detection.box
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            bw = box[2] - box[0]
            width_ratio = bw / frame_width
            center_ratio = cx / frame_width

            moving_in_from_left = (
                center_ratio < 0.48 and detection.lateral_velocity > 0.04
            )
            moving_in_from_right = (
                center_ratio > 0.52 and detection.lateral_velocity < -0.04
            )
            if width_ratio > 0.08 and (moving_in_from_left or moving_in_from_right):
                side_cut_risk = True

            if detection.label == "wrong way vehicle":
                wrong_side_risk = True

            if (
                center_lane_left < cx < center_lane_right
                and width_ratio > 0.18
                and cy > 0.5 * frame_height
                and detection.ttc_status == "critical_approach"
            ):
                frontal_conflict_risk = True
        return side_cut_risk, wrong_side_risk, frontal_conflict_risk

    def _compute_follow_distance_and_pinch(
        self, detections, frame_width, frame_height
    ):
        short_follow_distance = False
        pinch_point = False
        bus_blind_spot = False
        unsecured_load_risk = False
        if not detections:
            return (
                short_follow_distance,
                pinch_point,
                bus_blind_spot,
                unsecured_load_risk,
            )
        
        center_lane_left = frame_width * 0.30
        center_lane_right = frame_width * 0.70
        left_close = False
        right_close = False
        
        for detection in detections:
            box = detection.box
            label = detection.label
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            bw = box[2] - box[0]
            width_ratio = bw / frame_width
            bottom_ratio = box[3] / frame_height
            is_vehicle = label in (
                "bicycle",
                "car",
                "motorcycle",
                "bus",
                "truck",
                "rickshaw",
            )
            is_heavy = label in ("bus", "truck")
            
            if is_vehicle and center_lane_left < cx < center_lane_right:
                if width_ratio > 0.28 and bottom_ratio > 0.62:
                    short_follow_distance = True
            if is_vehicle and width_ratio > 0.20:
                if cx < frame_width * 0.35: left_close = True
                if cx > frame_width * 0.65: right_close = True
            if (
                is_heavy
                and 0.35 * frame_height < cy < 0.80 * frame_height
                and width_ratio > 0.20
            ):
                if 0.2 < cx / frame_width < 0.8: bus_blind_spot = True
            if label == "unsecured load":
                unsecured_load_risk = True
        
        if left_close and right_close: pinch_point = True
        return short_follow_distance, pinch_point, bus_blind_spot, unsecured_load_risk

    def _compute_entering_traffic_and_pedestrian(
        self, detections, frame_width, frame_height
    ):
        entering_traffic_risk = False
        pedestrian_crossing_risk = False
        if not detections:
            return entering_traffic_risk, pedestrian_crossing_risk
        
        for detection in detections:
            box = detection.box
            label = detection.label
            cx = (box[0] + box[2]) / 2
            bw = box[2] - box[0]
            bh = box[3] - box[1]
            h_ratio = bh / frame_height
            y_bottom_ratio = box[3] / frame_height
            x_center_ratio = cx / frame_width
            width_ratio = bw / frame_width
            
            if label == "person":
                close_and_central = (h_ratio > 0.25 and y_bottom_ratio > 0.7 and 0.15 < x_center_ratio < 0.85)
                crossing_motion = abs(detection.lateral_velocity) > 0.025
                if close_and_central and (
                    crossing_motion
                    or detection.ttc_status in ("closing_in", "critical_approach")
                ):
                    pedestrian_crossing_risk = True
            
            if label in ("car", "motorcycle", "rickshaw") and width_ratio > 0.12:
                extreme_edge = (cx < frame_width * 0.15 or cx > frame_width * 0.85)
                moving_toward_center = (
                    x_center_ratio < 0.5 and detection.lateral_velocity > 0.04
                ) or (
                    x_center_ratio > 0.5 and detection.lateral_velocity < -0.04
                )
                if (
                    extreme_edge
                    and moving_toward_center
                    and detection.ttc_status
                    in ("closing_in", "critical_approach")
                ):
                    entering_traffic_risk = True
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
        if not cap.isOpened():
            cap.release()
            raise ValueError(f"Could not open video: {self.video_path}")

        frame_data = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        source_fps = float(fps) if fps and fps > 0 else 30.0
        interval = max(
            int(round(source_fps / max(self.sampling_fps, 0.1))),
            1,
        )
        idx = 0
        previous_gray = None
        previous_timestamp = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % interval != 0:
                idx += 1
                continue

            timestamp = idx / source_fps
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if previous_gray is not None:
                elapsed = max(timestamp - previous_timestamp, 1 / source_fps)
                speed_status, is_erratic = self.estimate_speed_heuristic(
                    frame,
                    previous_gray,
                    elapsed,
                )
                self.total_frames_processed += 1
                frame_data.append(
                    self._analyze_single_frame(
                        frame,
                        idx,
                        timestamp,
                        speed_status,
                        is_erratic,
                    )
                )
                if self.max_frames and len(frame_data) >= self.max_frames:
                    break

            previous_gray = gray
            previous_timestamp = timestamp
            idx += 1

        cap.release()
        self.detector_metadata["inference_errors"] = dict(self.inference_errors)
        return frame_data

    def _analyze_single_frame(
        self, frame, frame_idx, timestamp, speed_status, is_erratic
    ):
        detections, inference_succeeded, errors = self.detector.predict(frame)
        self.inference_errors.update(errors)
        if not inference_succeeded:
            self.detection_failures += 1

        detections = self.tracker.update(
            detections,
            timestamp=timestamp,
            frame_width=frame.shape[1],
        )
        objects_detected = [detection.label for detection in detections]
        max_closeness = 0.0
        ttc_status = "stable"
        for detection in detections:
            x1, y1, x2, y2 = detection.box
            width_closeness = (x2 - x1) / (frame.shape[1] * 0.8)
            height_closeness = (y2 - y1) / (frame.shape[0] * 0.9)
            max_closeness = max(
                max_closeness,
                min(max(width_closeness, height_closeness), 1.0),
            )
            if detection.ttc_status == "critical_approach":
                ttc_status = "critical_approach"
            elif detection.ttc_status == "closing_in" and ttc_status == "stable":
                ttc_status = "closing_in"

        has_handheld, has_mounted, phone_risk = self.classify_phone_risk(
            detections, frame.shape[1], frame.shape[0]
        )
        is_glare = self.detect_glare(frame)
        is_night = self.detect_darkness_improved(frame)
        is_sandwich = self.check_sandwich_condition(
            detections, frame.shape[1]
        )
        side_cut, wrong_side, frontal_conflict = self._compute_dhaka_side_risks(
            detections, frame.shape[1], frame.shape[0]
        )
        short_follow, pinch, bus_blind, unsecured = (
            self._compute_follow_distance_and_pinch(
                detections, frame.shape[1], frame.shape[0]
            )
        )
        entering, ped_cross = self._compute_entering_traffic_and_pedestrian(
            detections, frame.shape[1], frame.shape[0]
        )
        wet_glare = self.detect_wet_or_glare_surface(frame, is_night)
        
        late_night_high_speed = (is_night and speed_status == "fast" and len(objects_detected) <= 1)

        return {
            "frame_id": frame_idx,
            "timestamp": timestamp,
            "frame": frame.copy(),
            "objects": objects_detected,
            "detections": [
                {
                    "box": [round(value, 2) for value in detection.box],
                    "label": detection.label,
                    "confidence": round(detection.confidence, 4),
                    "model_votes": detection.model_votes,
                    "track_id": detection.track_id,
                    "ttc_status": detection.ttc_status,
                    "ttc_seconds": (
                        round(detection.ttc_seconds, 3)
                        if detection.ttc_seconds is not None
                        else None
                    ),
                }
                for detection in detections
            ],
            "proximity_score": max_closeness,
            "ego_speed": speed_status,
            "is_erratic": is_erratic,
            "ttc_status": ttc_status,
            "glare": is_glare,
            "night": is_night,
            "phone_detected": has_handheld,
            "phone_mounted": has_mounted,
            "phone_risk": phone_risk,
            "sandwich_risk": is_sandwich,
            "side_cut_risk": side_cut,
            "wrong_side_risk": wrong_side,
            "frontal_conflict_risk": frontal_conflict,
            "short_follow_distance": short_follow,
            "pinch_point": pinch,
            "bus_blind_spot": bus_blind,
            "unsecured_load_risk": unsecured,
            "entering_traffic_risk": entering,
            "pedestrian_crossing_risk": ped_cross,
            "wet_or_glare_surface": wet_glare,
            "late_night_high_speed": late_night_high_speed,
        }
