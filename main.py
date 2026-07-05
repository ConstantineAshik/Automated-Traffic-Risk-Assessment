import sys
import os
import cv2

from core.config import PipelineConfig
from core.pipeline import analyze
from reporting.report_csv import write_predictions_csv
from reporting.report_text import write_report


def get_video_path():
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        if os.path.isfile(video_path):
            print(f"Using video from command-line: {video_path}\n")
            return video_path
        print(f"File not found: {video_path}\n")

    print("=" * 70)
    print("VIDEO PATH SELECTOR")
    print("=" * 70)
    print("\nEnter the path to your video file.")
    print("Supported formats: .mp4, .avi, .mov, .flv, .mkv\n")

    while True:
        video_path = input("Enter video path (or press Enter for default): ").strip()

        if not video_path:
            default_path = r"G:\Capstone c\rough_ride.mp4"
            if os.path.isfile(default_path):
                print(f"Using default video: {default_path}\n")
                return default_path
            print(f"Default path not found: {default_path}")
            print("Please enter a valid path.\n")
            continue

        if not os.path.isfile(video_path):
            print(f"File not found: {video_path}")
            print("Please check the path and try again.\n")
            continue

        valid_extensions = (".mp4", ".avi", ".mov", ".flv", ".mkv", ".webm", ".m4v")
        if not video_path.lower().endswith(valid_extensions):
            print(f"Invalid video format. Supported: {', '.join(valid_extensions)}")
            print("Please enter a valid video file.\n")
            continue

        print(f"Video selected: {video_path}\n")
        return video_path


def _save_frames(result):
    safe_frames_dir = os.path.join(os.getcwd(), "safe_frames")
    caution_frames_dir = os.path.join(os.getcwd(), "caution_frames")
    danger_frames_dir = os.path.join(os.getcwd(), "danger_frames")
    os.makedirs(safe_frames_dir, exist_ok=True)
    os.makedirs(caution_frames_dir, exist_ok=True)
    os.makedirs(danger_frames_dir, exist_ok=True)

    safe_count_saved = 0
    caution_count_saved = 0
    danger_count_saved = 0

    for i, smoothed in enumerate(result.smoothed_predictions):
        frame_info = result.raw_frame_data[i]
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
        except Exception as exc:
            print(f"Warning: could not save frame {frame_id}: {exc}")

    return safe_count_saved, caution_count_saved, danger_count_saved


def main():
    video_path = get_video_path()
    output_file = "ride_safety_report.txt"
    project_root = os.path.dirname(os.path.abspath(__file__))
    risk_model_path = os.path.join(project_root, "models", "risk_model.joblib")
    config = PipelineConfig(
        structured_risk_model_path=(
            risk_model_path if os.path.isfile(risk_model_path) else None
        )
    )

    print("=" * 70)
    print("DHAKA-RIDE SAFETY ANALYZER")
    print("=" * 70)
    print(f"\nProcessing: {video_path}\n")

    try:
        result = analyze(video_path, config)
    except Exception as exc:
        print(f"Failed to analyze video: {exc}")
        return

    try:
        preds_csv = write_predictions_csv(result)
        print(f"Frame-level predictions saved: {preds_csv}")
    except Exception as exc:
        print(f"Warning: failed to write predictions CSV: {exc}")

    saved_counts = _save_frames(result)
    print(
        f"Saved frames -> safe: {saved_counts[0]}, caution: {saved_counts[1]}, danger: {saved_counts[2]}"
    )

    write_report(result, output_file)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nReport saved: {output_file}")
    print("\nQuick Stats:")
    print(
        f"  Safe:    {result.safe_count}/{result.total_samples} "
        f"({result.safe_count/result.total_samples*100:.1f}%)"
    )
    print(
        f"  Caution: {result.caution_count}/{result.total_samples} "
        f"({result.caution_count/result.total_samples*100:.1f}%)"
    )
    print(
        f"  Danger:  {result.danger_count}/{result.total_samples} "
        f"({result.danger_count/result.total_samples*100:.1f}%)"
    )
    print(f"\n{result.verdict}")

    if not result.incomplete_analysis:
        print("\nDetection quality: GOOD")
    else:
        print(f"\nDetection quality: {100 - result.failure_rate:.1f}%")


if __name__ == "__main__":
    main()
