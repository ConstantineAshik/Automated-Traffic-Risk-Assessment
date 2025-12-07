import sys
import os
from collections import deque, Counter
from video_processor import VideoProcessor
from text_generator import TextGenerator
from risk_model import RiskModel
from risk_calculator import RiskCalculator
import cv2
import csv


def get_video_path():
    """
    Interactive video path selection with validation.
    Tries command-line arg, then user input, then defaults.
    """
    # Try command-line argument first
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        if os.path.isfile(video_path):
            print(f"✓ Using video from command-line: {video_path}\n")
            return video_path
        else:
            print(f"❌ File not found: {video_path}\n")
    
    # Interactive input
    print("=" * 70)
    print("VIDEO PATH SELECTOR")
    print("=" * 70)
    print("\nEnter the path to your video file.")
    print("Supported formats: .mp4, .avi, .mov, .flv, .mkv\n")
    
    while True:
        video_path = input("Enter video path (or press Enter for default): ").strip()
        
        # Use default if empty
        if not video_path:
            default_path = r"G:\Capstone c\rough_ride.mp4"
            if os.path.isfile(default_path):
                print(f"✓ Using default video: {default_path}\n")
                return default_path
            else:
                print(f"❌ Default path not found: {default_path}")
                print("Please enter a valid path.\n")
                continue
        
        # Validate path exists
        if not os.path.isfile(video_path):
            print(f"❌ File not found: {video_path}")
            print("Please check the path and try again.\n")
            continue
        
        # Validate video extension
        valid_extensions = ('.mp4', '.avi', '.mov', '.flv', '.mkv', '.webm', '.m4v')
        if not video_path.lower().endswith(valid_extensions):
            print(f"❌ Invalid video format. Supported: {', '.join(valid_extensions)}")
            print("Please enter a valid video file.\n")
            continue
        
        print(f"✓ Video selected: {video_path}\n")
        return video_path


def map_speed_category(speed_status):
    """Map optical flow speed to calculator category."""
    if speed_status == "stationary":
        return "stationary"
    elif speed_status == "slow":
        return "slow"
    elif speed_status == "fast":
        return "moderate"  # Optical flow "fast" = 30-40 km/h in Dhaka
    else:
        return "slow"


def suggest_actions(stats, verdict):
    """
    Returns actionable suggestions based on risk patterns and verdict.
    """
    suggestions = []
    
    if stats.get("Phone Distraction", 0) > 0:
        suggestions.append(
            "🚨 CRITICAL: Avoid using phone while riding. This is the #1 preventable risk."
        )
    
    if stats.get("High Speed Tailgating", 0) > 0:
        suggestions.append(
            "⚠️ Reduce speed and increase following distance to 4+ seconds behind vehicles."
        )
    
    if stats.get("Wrong Side Risk", 0) > 0:
        suggestions.append(
            "⚠️ When facing wrong-direction traffic, reduce speed and stay in predictable lane."
        )
    
    if stats.get("Side Cut Risk", 0) > 0:
        suggestions.append(
            "⚠️ Keep margin from road edges; expect sudden entries from rickshaws/motorcycles."
        )
    
    if stats.get("Pedestrian Crossing", 0) > 0:
        suggestions.append(
            "⚠️ Slow down near pedestrians and give them priority."
        )
    
    if stats.get("Bus Blind Spot", 0) > 0:
        suggestions.append(
            "⚠️ Avoid lingering beside buses/trucks; overtake decisively or fall back."
        )
    
    if stats.get("Late-Night High-Speed", 0) > 0:
        suggestions.append(
            "⚠️ On empty late-night roads, ride slower than headlight distance."
        )
    
    if stats.get("Wet Road / Glare", 0) > 0:
        suggestions.append(
            "⚠️ On wet/glare conditions, reduce speed and double your following distance."
        )
    
    # Positive reinforcement
    if verdict == "SAFE ✅" and not suggestions:
        suggestions.append(
            "✓ Excellent riding! Maintain current habits of awareness and safe spacing."
        )
    elif "MODERATE RISK" in verdict and len(suggestions) < 2:
        suggestions.append(
            "Focus on the identified hazard above. Most other riding is acceptable."
        )
    
    return suggestions


def main():
    """Main analysis pipeline."""
    
    # Get video path interactively
    video_path = get_video_path()
    
    output_file = "ride_safety_report.txt"
    
    print("=" * 70)
    print("DHAKA-RIDE SAFETY ANALYZER V9")
    print("=" * 70)
    print(f"\n📹 Processing: {video_path}\n")
    
    # Initialize components
    try:
        processor = VideoProcessor(video_path, window_size=10, danger_img_dir="danger_frames")
        text_gen = TextGenerator()
        risk_model = RiskModel()
        risk_calc = RiskCalculator()
    except Exception as e:
        print(f"❌ Failed to initialize components: {e}")
        return
    
    # Train model
    print("\n[1/5] Initializing Model...")
    try:
        risk_model.train_mock_model()
    except Exception as e:
        print(f"❌ Failed to train model: {e}")
        return
    
    # Process video
    print("\n[2/5] Processing Video...")
    try:
        raw_frame_data = processor.process_video()
    except Exception as e:
        print(f"❌ Error processing video: {e}")
        return
    
    if not raw_frame_data:
        print("❌ No frames found in video.")
        return
    
    # Check detection health
    failure_rate = (
        (processor.detection_failures / processor.total_frames_processed * 100)
        if processor.total_frames_processed > 0
        else 0
    )
    
    incomplete_analysis = failure_rate > 5
    print(f"✓ Processed {len(raw_frame_data)} frames")
    print(f"  Detection health: {100 - failure_rate:.1f}%")
    
    # Generate descriptions
    print("\n[3/5] Generating Context-Aware Descriptions...")
    try:
        descriptions = [text_gen.generate_description(f) for f in raw_frame_data]
        print(f"✓ Generated {len(descriptions)} descriptions")
    except Exception as e:
        print(f"❌ Failed to generate descriptions: {e}")
        return
    
    # Predict risks
    print("\n[4/5] Predicting Risk Levels (Improved Model)...")
    try:
        risk_predictions = risk_model.predict_risk(descriptions)
        print(f"✓ Got predictions")
        
        pred_dist = Counter(risk_predictions)
        print(f"  Distribution: {pred_dist[0]} SAFE, {pred_dist.get(1, 0)} CAUTION, {pred_dist.get(2, 0)} DANGER")
    except Exception as e:
        print(f"❌ Failed to predict risks: {e}")
        return
    
    # Apply fusion of numeric score and text-model, then temporal smoothing
    print("\n[5/5] Merging numeric scores with text-model and applying smoothing...")

    # 1) Compute numeric scores per frame
    numeric_scores = []
    for frame_data in raw_frame_data:
        speed_cat = map_speed_category(frame_data.get("ego_speed", "slow"))
        score = risk_calc.calculate_risk_score(frame_data, speed_cat)
        numeric_scores.append(score)

    # 2) Helper: merge labels (model_label:int, numeric_label:str) -> final int label
    def merge_labels(model_label, numeric_label):
        if numeric_label == "DANGER":
            return 2
        if numeric_label == "SAFE" and model_label == 0:
            return 0
        if model_label == 2 or numeric_label == "CAUTION":
            return 1
        return int(model_label)

    # 3) Compute merged label per frame
    merged_predictions = []
    numeric_label_list = []
    for model_lab, score in zip(risk_predictions, numeric_scores):
        nlabel = risk_calc.score_to_label(score)
        numeric_label_list.append(nlabel)
        merged = merge_labels(int(model_lab), nlabel)
        merged_predictions.append(merged)

    # 4) Temporal smoothing (keep max in window to preserve high-risk events)
    window_size = 5
    risk_window = deque(maxlen=window_size)
    smoothed_predictions = []
    for lab in merged_predictions:
        risk_window.append(int(lab))
        smoothed_predictions.append(max(risk_window))

    print(f"✓ Computed numeric scores and applied {window_size}-frame smoothing window")
    
    # ============ WRITE FRAME-LEVEL MODEL PREDICTIONS CSV ============
    # Columns: frame_id, model_label, numeric_label, merged_label, description, numeric_score, phone_risk, smoothed_label
    preds_csv = os.path.join(os.getcwd(), "model_predictions.csv")
    try:
        with open(preds_csv, "w", newline="", encoding="utf-8") as cf:
            writer = csv.writer(cf)
            writer.writerow(["frame_id", "model_label", "numeric_label", "merged_label", "description", "numeric_score", "phone_risk", "smoothed_label"])
            for i, (desc, model_lab, nlabel, merged, score) in enumerate(zip(descriptions, risk_predictions, numeric_label_list, merged_predictions, numeric_scores)):
                frame_id = raw_frame_data[i].get("frame_id", i)
                phone_risk = raw_frame_data[i].get("phone_risk", "")
                writer.writerow([int(frame_id), int(model_lab), nlabel, int(merged), desc, f"{score:.2f}", phone_risk, int(smoothed_predictions[i])])
        print(f"✓ Frame-level predictions saved: {preds_csv}")
    except Exception as e:
        print(f"⚠️ Failed to write predictions CSV: {e}")
    
    # ============ SAVE ANALYZED FRAMES INTO FOLDERS ============
    safe_frames_dir = os.path.join(os.getcwd(), "safe_frames")
    caution_frames_dir = os.path.join(os.getcwd(), "caution_frames")
    danger_frames_dir = os.path.join(os.getcwd(), "danger_frames")
    os.makedirs(safe_frames_dir, exist_ok=True)
    os.makedirs(caution_frames_dir, exist_ok=True)
    os.makedirs(danger_frames_dir, exist_ok=True)
    
    safe_count_saved = 0
    caution_count_saved = 0
    danger_count_saved = 0
    
    for i, (desc, model_pred, smoothed) in enumerate(zip(descriptions, risk_predictions, smoothed_predictions)):
        frame_info = raw_frame_data[i]
        frame = frame_info.get("frame")
        frame_id = frame_info.get("frame_id", i)
        if frame is None:
            continue
        filename = f"frame_{int(frame_id):06d}.jpg"
        try:
            if smoothed == 0:
                cv2.imwrite(os.path.join(safe_frames_dir, filename), frame)
                safe_count_saved += 1
            elif smoothed == 1:
                cv2.imwrite(os.path.join(caution_frames_dir, filename), frame)
                caution_count_saved += 1
            else:
                cv2.imwrite(os.path.join(danger_frames_dir, filename), frame)
                danger_count_saved += 1
        except Exception as e:
            # continue on error but log minimal info
            print(f"⚠️ Could not save frame {frame_id}: {e}")
    
    print(f"Saved frames → safe: {safe_count_saved}, caution: {caution_count_saved}, danger: {danger_count_saved}")
    
    # Make saved counts available for the report
    saved_counts = (safe_count_saved, caution_count_saved, danger_count_saved)
    
    # Generate report
    print("\n[6/6] Generating Report...\n")
    
    # ============ STATISTICS COLLECTION ============
    
    stats = {
        "High Speed Tailgating": 0,
        "Phone Distraction (5+ frames)": 0,  # NEW: only real distraction
        "Phone Usage Caution (brief)": 0,    # NEW: brief checks
        "Glare Blindness": 0,
        "Side Cut Risk": 0,
        "Wrong Side Risk": 0,
        "Short Follow Distance": 0,
        "Pinch Point": 0,
        "Bus Blind Spot": 0,
        "Unsecured Load": 0,
        "Entering Traffic Conflict": 0,
        "Pedestrian Crossing": 0,
        "Wet Road / Glare": 0,
        "Late-Night High-Speed": 0,
    }
    
    safe_count = 0
    caution_count = 0
    danger_count = 0

    # Additional safety checks (hard rules)
    max_score = max(numeric_scores) if numeric_scores else 0
    phone_danger_frames = sum(1 for f in raw_frame_data if f.get("phone_risk") == "danger")
    # count longest continuous DANGER run (max_run)
    max_run = 0
    current_run = 0
    for r in smoothed_predictions:
        if r == 2:
            current_run += 1
            if current_run > max_run:
                max_run = current_run
        else:
            current_run = 0

    # sliding-window critical episode detection (5 seconds window ~10 samples at 2fps)
    episode_count = 0
    window_len = 10
    for i in range(0, len(smoothed_predictions), window_len):
        window = smoothed_predictions[i:i+window_len]
        if not window:
            continue
        if sum(1 for r in window if r == 2) >= len(window) // 2:
            episode_count += 1
    
    # ============ FRAME ANALYSIS & REPORT GENERATION ============
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("DHAKA-RIDE SAFETY ANALYZER V9 (IMPROVED)\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Video Analyzed: {video_path}\n\n")
        
        f.write("This analysis uses:\n")
        f.write("✓ Speed-aware proximity interpretation\n")
        f.write("✓ Traffic jam exception handling\n")
        f.write("✓ Proper phone detection weighting\n")
        f.write("✓ Bangladesh-specific traffic context\n")
        f.write("✓ Temporal smoothing for noise reduction\n\n")
        
        if incomplete_analysis:
            f.write(f"⚠️ WARNING: Detection failure rate {failure_rate:.1f}%\n")
            f.write("Analysis may be incomplete. Manual review recommended.\n\n")
        
        # Frame-by-frame log
        f.write("FRAME-BY-FRAME LOG (CAUTION+ EVENTS ONLY)\n")
        f.write("-" * 70 + "\n")
        
        for i, (desc, merged_pred, smoothed_risk) in enumerate(
            zip(descriptions, merged_predictions, smoothed_predictions)
        ):
            risk_label = risk_model.interpret_risk(smoothed_risk)
            frame_id = raw_frame_data[i]["frame_id"]
            speed = raw_frame_data[i].get("ego_speed", "unknown")
            numeric_score = numeric_scores[i]
            
            # NEW: Count phone risk with temporal tracking
            phone_risk = raw_frame_data[i].get("phone_risk", "safe")
            if phone_risk == "danger":
                stats["Phone Distraction (5+ frames)"] += 1
            elif phone_risk == "caution":
                stats["Phone Usage Caution (brief)"] += 1
            
            # Count other risk statistics
            if "high_speed_tailgating" in desc:
                stats["High Speed Tailgating"] += 1
            if "glare_blindness" in desc:
                stats["Glare Blindness"] += 1
            if "side_cut_risk" in desc:
                stats["Side Cut Risk"] += 1
            if "wrong_side_risk" in desc:
                stats["Wrong Side Risk"] += 1
            if "short_follow_distance" in desc:
                stats["Short Follow Distance"] += 1
            if "pinch_point" in desc:
                stats["Pinch Point"] += 1
            if "bus_blind_spot" in desc:
                stats["Bus Blind Spot"] += 1
            if "unsecured_truck_load" in desc:
                stats["Unsecured Load"] += 1
            if "entering_traffic_conflict" in desc:
                stats["Entering Traffic Conflict"] += 1
            if "pedestrian_crossing" in desc:
                stats["Pedestrian Crossing"] += 1
            if "wet_road_glare" in desc:
                stats["Wet Road / Glare"] += 1
            if "late_night_high_speed" in desc:
                stats["Late-Night High-Speed"] += 1
            
            # Count classifications
            if smoothed_risk == 0:
                safe_count += 1
            elif smoothed_risk == 1:
                caution_count += 1
            else:
                danger_count += 1
            
            # Log CAUTION+ events
            if smoothed_risk >= 1:
                line = f"[Frame {frame_id:5d}] {risk_label:7s} ({speed:10s}) | Score: {numeric_score:3.0f}/100 | {desc[:45]}\n"
                f.write(line)
        
        # Summary statistics
        total_samples = len(descriptions)
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("=" * 70 + "\n")
        f.write(f"Total Frames Analyzed: {total_samples}\n")
        f.write(f"Safe Frames: {safe_count} ({safe_count/total_samples*100:.1f}%)\n")
        f.write(f"Caution Frames: {caution_count} ({caution_count/total_samples*100:.1f}%)\n")
        f.write(f"Danger Frames: {danger_count} ({danger_count/total_samples*100:.1f}%)\n\n")
        
        # Risk factor breakdown
        f.write("DETECTED RISK FACTORS\n")
        f.write("-" * 70 + "\n")
        
        for factor, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                f.write(f"  {factor}: {count}\n")
        
        if all(c == 0 for c in stats.values()):
            f.write("  (No specific risk factors detected)\n")
        
        # Verdict logic (improved with hard safety rules + numeric-model fusion)
        f.write("\n" + "=" * 70 + "\n")
        f.write("FINAL VERDICT\n")
        f.write("=" * 70 + "\n")

        danger_pct = (danger_count / total_samples * 100) if total_samples > 0 else 0
        caution_pct = (caution_count / total_samples * 100) if total_samples > 0 else 0

        # Start conservative: assume SAFE unless a hard rule applies
        verdict = "SAFE ✅"
        reason = f"Good riding with {100-caution_pct-danger_pct:.1f}% safe frames. Keep it up!"

        # Hard rules and overrides (Dhaka-tuned)
        if phone_danger_frames > 0 or stats.get("Phone Distraction (5+ frames)", 0) > 0:
            verdict = "UNSAFE ❌"
            reason = "Active phone distraction detected (danger). Immediate corrective action required."
        elif max_score >= 95 or max_run >= 3:
            verdict = "DANGER ❌ – Serious risk episodes detected"
            reason = "Sustained consecutive danger events or extreme single-frame risk detected. Immediate attention required."
        elif (danger_count / total_samples) > 0.03 or max_score >= 80 or episode_count >= 1:
            verdict = "MODERATE RISK ⚠️"
            reason = "Noticeable risky behavior detected. Reduce speed and increase distance; review high-risk segments."
        elif caution_pct > 20:
            verdict = "MODERATE RISK ⚠️"
            reason = f"Frequent minor risk indicators ({caution_pct:.1f}%). Stay alert and reduce risky maneuvers."

        # Additional nudges: presence of wrong-side or pedestrian critical risks increases severity
        if verdict == "SAFE ✅":
            if stats.get("Wrong Side Risk", 0) > 0 and max_score >= 70:
                verdict = "MODERATE RISK ⚠️"
                reason = "Wrong-side interactions detected alongside high scores; be cautious."
            if stats.get("Pedestrian Crossing", 0) > 0 and max_score >= 65:
                verdict = "MODERATE RISK ⚠️"
                reason = "Pedestrian interactions with high approach scores; reduce speed near pedestrians."

        f.write(f"\n{verdict}\n\n")
        f.write(f"Reason:\n{reason}\n")
        
        # Recommendations
        f.write("\n" + "=" * 70 + "\n")
        f.write("RECOMMENDED ACTIONS\n")
        f.write("=" * 70 + "\n")
        
        suggestions = suggest_actions(stats, verdict)
        for i, suggestion in enumerate(suggestions, 1):
            f.write(f"\n{i}. {suggestion}\n")
        
        # Health warning
        if incomplete_analysis:
            f.write("\n" + "=" * 70 + "\n")
            f.write("⚠️ ANALYSIS HEALTH WARNING\n")
            f.write("=" * 70 + "\n")
            f.write(f"Detection failed on {processor.detection_failures} out of ")
            f.write(f"{processor.total_frames_processed} frames ({failure_rate:.1f}%).\n")
            f.write("Video quality issues may affect accuracy.\n")
    
    # Console output
    print("\n" + "=" * 70)
    print("✓ ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\n📄 Report saved: {output_file}")
    print(f"\nQuick Stats:")
    print(f"  Safe:    {safe_count}/{total_samples} ({safe_count/total_samples*100:.1f}%)")
    print(f"  Caution: {caution_count}/{total_samples} ({caution_count/total_samples*100:.1f}%)")
    print(f"  Danger:  {danger_count}/{total_samples} ({danger_count/total_samples*100:.1f}%)")
    print(f"\n{verdict}")
    
    if not incomplete_analysis:
        print("\n✓ Detection quality: GOOD")
    else:
        print(f"\n⚠️ Detection quality: {100-failure_rate:.1f}%")
if __name__ == "__main__":
    main()