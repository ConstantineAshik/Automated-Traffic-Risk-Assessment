from typing import Dict, List

from core.types import AnalysisResult


def _suggest_actions(stats: Dict[str, int], verdict: str) -> List[str]:
    suggestions = []

    if stats.get("Phone Distraction (sustained)", 0) > 0:
        suggestions.append(
            "CRITICAL: Avoid using phone while riding. This is the top preventable risk."
        )
    if stats.get("High Speed Tailgating", 0) > 0:
        suggestions.append(
            "Reduce speed and increase following distance to 4+ seconds behind vehicles."
        )
    if stats.get("Wrong Side Risk", 0) > 0:
        suggestions.append(
            "When facing wrong-direction traffic, reduce speed and stay in a predictable lane."
        )
    if stats.get("Side Cut Risk", 0) > 0:
        suggestions.append(
            "Keep margin from road edges; expect sudden entries from rickshaws/motorcycles."
        )
    if stats.get("Pedestrian Crossing", 0) > 0:
        suggestions.append("Slow down near pedestrians and give them priority.")
    if stats.get("Bus Blind Spot", 0) > 0:
        suggestions.append("Avoid lingering beside buses/trucks; overtake decisively or fall back.")
    if stats.get("Late-Night High-Speed", 0) > 0:
        suggestions.append(
            "On empty late-night roads, ride slower than headlight distance."
        )
    if stats.get("Wet Road / Glare", 0) > 0:
        suggestions.append(
            "On wet/glare conditions, reduce speed and double your following distance."
        )

    if verdict == "SAFE" and not suggestions:
        suggestions.append("Excellent riding. Maintain current habits of awareness and safe spacing.")
    elif verdict in ("CAUTION", "CAUTION_WITH_DANGER_MOMENT", "MODERATE") and len(suggestions) < 2:
        suggestions.append("Focus on the identified hazard above. Most other riding is acceptable.")

    return suggestions


def _display_verdict(verdict: str) -> str:
    return str(verdict).replace("_", " ").title()


def write_report(result: AnalysisResult, output_file: str) -> None:
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("DHAKA-RIDE SAFETY ANALYZER\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Video Analyzed: {result.video_path}\n\n")
        loaded_models = result.detector_metadata.get("loaded_models", [])
        f.write(
            "Detector Models: "
            + (", ".join(loaded_models) if loaded_models else "(none)")
            + "\n\n"
        )
        f.write("This analysis uses:\n")
        f.write("- COCO object detection for forward-path obstacles\n")
        f.write("- TTC-aware proximity interpretation\n")
        f.write("- Traffic jam and stable-distance exception handling\n")
        f.write("- Phone detection weighting\n")
        f.write("- Motion context from tracking and optical flow\n")
        f.write("- Temporal smoothing for noise reduction\n\n")

        if result.incomplete_analysis:
            f.write(f"WARNING: Detection failure rate {result.failure_rate:.1f}%\n")
            f.write("Analysis may be incomplete. Manual review recommended.\n\n")

        f.write("FRAME-BY-FRAME LOG (CAUTION+ EVENTS ONLY)\n")
        f.write("-" * 70 + "\n")

        for i, (desc, numeric_label) in enumerate(
            zip(result.descriptions, result.numeric_labels)
        ):
            if numeric_label == "SAFE":
                continue
            frame_id = result.raw_frame_data[i].get("frame_id", i)
            speed = result.raw_frame_data[i].get("ego_speed", "unknown")
            score = result.numeric_scores[i]
            line = (
                f"[Frame {frame_id:5d}] {numeric_label:7s} ({speed:10s}) | "
                f"Score: {score:3.0f}/100 | {desc[:45]}\n"
            )
            f.write(line)

        total = result.total_samples or 1
        f.write("\n" + "=" * 70 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("=" * 70 + "\n")
        f.write(f"Total Frames Analyzed: {result.total_samples}\n")
        f.write(f"Safe Frames: {result.safe_count} ({result.safe_count/total*100:.1f}%)\n")
        f.write(f"Caution Frames: {result.caution_count} ({result.caution_count/total*100:.1f}%)\n")
        f.write(f"Danger Frames: {result.danger_count} ({result.danger_count/total*100:.1f}%)\n\n")

        f.write("DETECTED RISK FACTORS\n")
        f.write("-" * 70 + "\n")
        for factor, count in sorted(result.stats.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                f.write(f"  {factor}: {count}\n")
        if all(c == 0 for c in result.stats.values()):
            f.write("  (No specific risk factors detected)\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("FINAL VERDICT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{_display_verdict(result.verdict)}\n\n")
        f.write("Reason:\n")
        f.write(f"{result.verdict_reason}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("RECOMMENDED ACTIONS\n")
        f.write("=" * 70 + "\n")
        suggestions = _suggest_actions(result.stats, result.verdict)
        for i, suggestion in enumerate(suggestions, 1):
            f.write(f"\n{i}. {suggestion}\n")

        if result.incomplete_analysis:
            f.write("\n" + "=" * 70 + "\n")
            f.write("ANALYSIS HEALTH WARNING\n")
            f.write("=" * 70 + "\n")
            f.write(
                f"Detection failed on {int(result.failure_rate)}% of frames.\n"
            )
            f.write("Video quality issues may affect accuracy.\n")
