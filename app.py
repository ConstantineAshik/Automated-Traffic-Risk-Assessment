import hashlib
import html
import os
import tempfile
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from core.config import PipelineConfig
from core.pipeline import analyze


st.set_page_config(
    page_title=" Traffic Risk Analysis",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
<style>
    :root {
        --ink: #f8fafc;
        --muted: #94a3b8;
        --line: rgba(148, 163, 184, 0.18);
        --panel: rgba(15, 23, 42, 0.72);
        --accent: #22d3ee;
        --safe: #34d399;
        --caution: #fbbf24;
        --danger: #fb7185;
    }

    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(34, 211, 238, 0.12), transparent 30rem),
            radial-gradient(circle at 95% 15%, rgba(99, 102, 241, 0.12), transparent 28rem),
            #070b14;
        color: var(--ink);
    }

    .block-container {
        max-width: 1280px;
        padding-top: 2.25rem;
        padding-bottom: 4rem;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .hero {
        padding: 1.2rem 0 1.8rem;
    }

    .eyebrow {
        color: var(--accent);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
    }

    .hero h1 {
        color: var(--ink);
        font-size: clamp(2.4rem, 5vw, 4.8rem);
        line-height: 0.98;
        letter-spacing: -0.055em;
        margin: 0.55rem 0 1rem;
        max-width: 920px;
    }

    .hero p {
        color: var(--muted);
        font-size: 1.05rem;
        line-height: 1.7;
        max-width: 720px;
    }

    .status-pill {
        align-items: center;
        background: rgba(52, 211, 153, 0.1);
        border: 1px solid rgba(52, 211, 153, 0.28);
        border-radius: 999px;
        color: #a7f3d0;
        display: inline-flex;
        font-size: 0.8rem;
        gap: 0.5rem;
        margin-top: 0.7rem;
        padding: 0.42rem 0.75rem;
    }

    .status-dot {
        background: var(--safe);
        border-radius: 50%;
        box-shadow: 0 0 0 4px rgba(52, 211, 153, 0.12);
        height: 0.45rem;
        width: 0.45rem;
    }

    [data-testid="stFileUploader"] {
        background: rgba(15, 23, 42, 0.64);
        border: 1px dashed rgba(34, 211, 238, 0.35);
        border-radius: 18px;
        padding: 0.65rem;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: rgba(2, 6, 23, 0.35);
        border: 0;
        border-radius: 13px;
    }

    div.stButton > button {
        border-radius: 10px;
        font-weight: 700;
        min-height: 2.8rem;
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(120deg, #0891b2, #4f46e5);
        border: 0;
        box-shadow: 0 12px 30px rgba(8, 145, 178, 0.2);
    }

    [data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 15px;
        min-height: 132px;
        padding: 1.15rem 1.2rem;
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted);
    }

    [data-testid="stMetricValue"] {
        color: var(--ink);
        letter-spacing: -0.04em;
    }

    .section-kicker {
        color: var(--accent);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        margin-bottom: 0.25rem;
        text-transform: uppercase;
    }

    .section-title {
        color: var(--ink);
        font-size: 1.55rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        margin-bottom: 1rem;
    }

    .verdict {
        border: 1px solid var(--line);
        border-left-width: 5px;
        border-radius: 16px;
        margin: 0.6rem 0 1.35rem;
        padding: 1.3rem 1.4rem;
    }

    .verdict.safe {
        background: rgba(52, 211, 153, 0.08);
        border-left-color: var(--safe);
    }

    .verdict.moderate {
        background: rgba(251, 191, 36, 0.08);
        border-left-color: var(--caution);
    }

    .verdict.danger {
        background: rgba(251, 113, 133, 0.08);
        border-left-color: var(--danger);
    }

    .verdict-label {
        color: var(--muted);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.13em;
        text-transform: uppercase;
    }

    .verdict-value {
        color: var(--ink);
        font-size: 1.55rem;
        font-weight: 800;
        margin: 0.18rem 0 0.25rem;
    }

    .verdict-reason {
        color: #cbd5e1;
        line-height: 1.55;
    }

    .evidence-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid var(--line);
        border-radius: 13px;
        margin: -0.25rem 0 1rem;
        padding: 0.85rem 1rem;
    }

    .evidence-meta {
        color: var(--accent);
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }

    .evidence-title {
        color: var(--ink);
        font-size: 1rem;
        font-weight: 700;
        margin: 0.22rem 0;
    }

    .evidence-copy {
        color: var(--muted);
        font-size: 0.88rem;
        line-height: 1.5;
    }

    hr {
        border-color: var(--line) !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


LABELS = {0: "Safe", 1: "Caution", 2: "Danger"}


def _readable_description(description):
    return " · ".join(token.replace("_", " ").title() for token in description.split())


def _video_metadata(video_path):
    capture = cv2.VideoCapture(video_path)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": (frame_count / fps) if fps > 0 else 0,
        "resolution": f"{width} x {height}" if width and height else "Unknown",
    }


def _format_time(seconds):
    minutes, secs = divmod(max(0, int(round(seconds))), 60)
    return f"{minutes:02d}:{secs:02d}"


def run_analysis(video_path):
    """Run the real pipeline and shape its output for the interactive dashboard."""
    metadata = _video_metadata(video_path)
    risk_model_path = Path(__file__).resolve().parent / "models" / "risk_model.joblib"
    config = PipelineConfig(
        structured_risk_model_path=(
            str(risk_model_path) if risk_model_path.is_file() else None
        )
    )
    result = analyze(video_path, config)
    total = result.total_samples or 1
    fps = metadata["fps"]

    timeline = []
    sample_rows = []
    for index, (frame_data, score, prediction, description) in enumerate(
        zip(
            result.raw_frame_data,
            result.numeric_scores,
            result.smoothed_predictions,
            result.descriptions,
        )
    ):
        frame_id = int(frame_data.get("frame_id", index))
        seconds = (frame_id / fps) if fps > 0 else (index / config.sampling_fps)
        level = LABELS.get(int(prediction), "Unknown")
        objects = ", ".join(dict.fromkeys(frame_data.get("objects", []))) or "None detected"
        readable_description = _readable_description(description)

        timeline.append(
            {
                "Sample": index + 1,
                "Time (seconds)": round(seconds, 2),
                "Risk score": int(score),
            }
        )
        sample_rows.append(
            {
                "Sample": index + 1,
                "Time": _format_time(seconds),
                "Frame": frame_id,
                "Level": level,
                "Score": int(score),
                "Speed": str(frame_data.get("ego_speed", "unknown")).title(),
                "Objects": objects,
                "Detected conditions": readable_description,
            }
        )

    flagged_indices = [
        index
        for index, prediction in enumerate(result.smoothed_predictions)
        if int(prediction) > 0
    ]
    flagged_indices.sort(
        key=lambda index: (result.numeric_scores[index], result.smoothed_predictions[index]),
        reverse=True,
    )

    evidence_frames = []
    for index in flagged_indices[:12]:
        frame_data = result.raw_frame_data[index]
        frame_id = int(frame_data.get("frame_id", index))
        seconds = (frame_id / fps) if fps > 0 else (index / config.sampling_fps)
        evidence_frames.append(
            {
                "frame": cv2.cvtColor(frame_data["frame"], cv2.COLOR_BGR2RGB),
                "frame_id": frame_id,
                "time": _format_time(seconds),
                "level": LABELS.get(int(result.smoothed_predictions[index]), "Unknown"),
                "description": _readable_description(result.descriptions[index]),
                "score": int(result.numeric_scores[index]),
                "objects": ", ".join(dict.fromkeys(frame_data.get("objects", [])))
                or "No relevant objects detected",
            }
        )

    counts = {
        "Safe": result.safe_count,
        "Caution": result.caution_count,
        "Danger": result.danger_count,
    }
    return {
        "total_samples": result.total_samples,
        "counts": counts,
        "safe_percentage": (result.safe_count / total) * 100,
        "danger_percentage": (result.danger_count / total) * 100,
        "average_score": sum(result.numeric_scores) / total,
        "max_score": result.max_score,
        "timeline": timeline,
        "samples": sample_rows,
        "evidence_frames": evidence_frames,
        "risk_factors": dict(result.stats),
        "verdict": result.verdict,
        "verdict_reason": result.verdict_reason,
        "failure_rate": result.failure_rate,
        "incomplete_analysis": result.incomplete_analysis,
        "max_danger_run": result.max_run,
        "danger_episodes": result.episode_count,
        "phone_danger_frames": result.phone_danger_frames,
        "metadata": metadata,
        "detector_metadata": result.detector_metadata,
    }


def _section(kicker, title):
    st.markdown(
        f'<div class="section-kicker">{html.escape(kicker)}</div>'
        f'<div class="section-title">{html.escape(title)}</div>',
        unsafe_allow_html=True,
    )


def render_results(data):
    st.divider()
    _section("Live analysis output", "What the model found in this video")
    st.caption(
        "These results were calculated from the uploaded footage. "
        "No sample, demo, or pre-filled result data is used."
    )

    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "Safety rate",
        f"{data['safe_percentage']:.1f}%",
        help="Share of analyzed samples classified as safe after smoothing.",
    )
    metric_columns[1].metric(
        "Peak risk",
        f"{data['max_score']}/100",
        help="Highest numeric risk score found in an analyzed sample.",
    )
    metric_columns[2].metric(
        "Danger samples",
        data["counts"]["Danger"],
        f"{data['danger_percentage']:.1f}% of samples",
        delta_color="inverse",
    )
    metric_columns[3].metric(
        "Samples analyzed",
        data["total_samples"],
        help="Frames sampled by the analysis pipeline, not the video's total frame count.",
    )

    verdict_class = (
        "safe"
        if data["verdict"] == "SAFE"
        else "moderate"
        if data["verdict"] == "MODERATE"
        else "danger"
    )
    st.markdown(
        f"""
        <div class="verdict {verdict_class}">
            <div class="verdict-label">Final assessment</div>
            <div class="verdict-value">{html.escape(data["verdict"])}</div>
            <div class="verdict-reason">{html.escape(data["verdict_reason"])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if data["incomplete_analysis"]:
        st.warning(
            f"Object detection failed on {data['failure_rate']:.1f}% of analyzed samples. "
            "Treat this result as incomplete."
        )

    chart_column, distribution_column = st.columns([1.75, 1], gap="large")
    with chart_column:
        _section("Timeline", "Risk score across the ride")
        timeline_frame = pd.DataFrame(data["timeline"])
        st.line_chart(
            timeline_frame,
            x="Time (seconds)",
            y="Risk score",
            height=320,
            color="#22d3ee",
        )
        st.caption("Numeric score: Safe 0–24 · Caution 25–54 · Danger 55–100")

    with distribution_column:
        _section("Distribution", "Classification mix")
        distribution = pd.DataFrame(
            {
                "Level": ["Safe", "Caution", "Danger"],
                "Samples": [
                    data["counts"]["Safe"],
                    data["counts"]["Caution"],
                    data["counts"]["Danger"],
                ],
            }
        )
        st.bar_chart(
            distribution,
            x="Level",
            y="Samples",
            height=320,
            color="#6366f1",
        )

    st.divider()
    insight_column, quality_column = st.columns([1.45, 1], gap="large")
    with insight_column:
        _section("Detected signals", "Risk factors found")
        active_factors = [
            (name, count)
            for name, count in data["risk_factors"].items()
            if count > 0
        ]
        active_factors.sort(key=lambda item: item[1], reverse=True)
        if active_factors:
            for name, count in active_factors:
                st.progress(
                    min(count / max(data["total_samples"], 1), 1.0),
                    text=f"{name} · {count} sample{'s' if count != 1 else ''}",
                )
        else:
            st.success("No configured risk factors were detected in the analyzed samples.")

    with quality_column:
        _section("Run details", "Analysis at a glance")
        detail_columns = st.columns(2)
        detail_columns[0].metric("Average risk", f"{data['average_score']:.1f}/100")
        detail_columns[1].metric("Longest danger run", data["max_danger_run"])
        detail_columns[0].metric("Danger episodes", data["danger_episodes"])
        detail_columns[1].metric("Detection failures", f"{data['failure_rate']:.1f}%")
        metadata = data["metadata"]
        detector_metadata = data["detector_metadata"]
        st.caption(
            f"Video: {_format_time(metadata['duration_seconds'])} · "
            f"{metadata['resolution']} · {metadata['fps']:.1f} FPS"
        )
        st.caption(
            "Detector ensemble: "
            + ", ".join(detector_metadata.get("loaded_models", []))
        )

    st.divider()
    _section("Visual evidence", "Highest-risk flagged samples")
    if data["evidence_frames"]:
        st.caption(
            "Frames below come directly from the uploaded video and are ordered by numeric risk score."
        )
        evidence_columns = st.columns(2, gap="large")
        for index, evidence in enumerate(data["evidence_frames"]):
            with evidence_columns[index % 2]:
                st.image(evidence["frame"], use_container_width=True)
                st.markdown(
                    f"""
                    <div class="evidence-card">
                        <div class="evidence-meta">
                            {html.escape(evidence["level"])} · {html.escape(evidence["time"])}
                            · Frame {evidence["frame_id"]}
                        </div>
                        <div class="evidence-title">Risk score {evidence["score"]}/100</div>
                        <div class="evidence-copy">
                            {html.escape(evidence["description"])}<br>
                            Objects: {html.escape(evidence["objects"])}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.success("The pipeline did not classify any analyzed sample as caution or danger.")

    st.divider()
    _section("Sample log", "Every analyzed observation")
    st.caption("This table is the on-screen analysis record generated for this upload.")
    sample_frame = pd.DataFrame(data["samples"])
    st.dataframe(
        sample_frame,
        use_container_width=True,
        hide_index=True,
        height=min(520, 38 + (len(sample_frame) * 35)),
        column_config={
            "Score": st.column_config.ProgressColumn(
                "Score",
                help="Numeric risk score for this sampled frame.",
                min_value=0,
                max_value=100,
                format="%d",
            ),
        },
    )


if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "analyzed_upload_id" not in st.session_state:
    st.session_state.analyzed_upload_id = None


st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Computer vision safety analysis</div>
        <h1>See the risk inside every ride.</h1>
        <p>
            Upload road footage to inspect speed context, proximity, distraction,
            traffic conflicts, and environmental hazards—then review the findings
            directly on screen.
        </p>
        <div class="status-pill">
            <span class="status-dot"></span>
            Local analysis · Your video stays on this machine
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

upload_column, explainer_column = st.columns([1.6, 1], gap="large")
with upload_column:
    uploaded_file = st.file_uploader(
        "Choose riding footage",
        type=["mp4", "avi", "mov"],
        help="Supported formats: MP4, AVI, and MOV.",
    )
with explainer_column:
    st.info(
        "**What you will see**\n\n"
        "A final safety assessment, risk timeline, detected factors, "
        "flagged source frames, and the complete analyzed-sample log."
    )

if uploaded_file is not None:
    video_bytes = uploaded_file.getvalue()
    upload_id = hashlib.sha256(video_bytes).hexdigest()

    if (
        st.session_state.analyzed_upload_id is not None
        and st.session_state.analyzed_upload_id != upload_id
    ):
        st.session_state.analysis_result = None
        st.session_state.analyzed_upload_id = None

    st.video(video_bytes)
    action_column, note_column = st.columns([0.8, 2.2], vertical_alignment="center")
    with action_column:
        analyze_clicked = st.button(
            "Analyze this video",
            type="primary",
            use_container_width=True,
        )
    with note_column:
        st.caption(
            f"{uploaded_file.name} · {len(video_bytes) / (1024 * 1024):.1f} MB · "
            "Results will appear below and stay visible while this upload is selected."
        )

    if analyze_clicked:
        suffix = Path(uploaded_file.name).suffix or ".mp4"
        video_path = None
        status = st.status("Analyzing uploaded footage…", expanded=True)
        try:
            status.write("Preparing the video for frame sampling")
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
                temporary_file.write(video_bytes)
                video_path = temporary_file.name

            status.write("Running object detection and traffic-risk scoring")
            analysis_data = run_analysis(video_path)
            st.session_state.analysis_result = analysis_data
            st.session_state.analyzed_upload_id = upload_id
            status.update(
                label=f"Analysis complete · {analysis_data['total_samples']} samples reviewed",
                state="complete",
                expanded=False,
            )
        except Exception as exc:
            st.session_state.analysis_result = None
            st.session_state.analyzed_upload_id = None
            status.update(label="Analysis could not be completed", state="error")
            st.error(f"Analysis error: {exc}")
        finally:
            if video_path and os.path.exists(video_path):
                os.remove(video_path)

    if (
        st.session_state.analysis_result is not None
        and st.session_state.analyzed_upload_id == upload_id
    ):
        render_results(st.session_state.analysis_result)
else:
    st.markdown("---")
    st.caption("Upload a video to begin. Results are never populated until an analysis finishes.")
