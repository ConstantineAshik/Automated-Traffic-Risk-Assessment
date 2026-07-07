import hashlib
import html
import io
import os
import tempfile
import zipfile
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from core.config import PipelineConfig
from core.pipeline import analyze


st.set_page_config(
    page_title="Automated Traffic Risk Assessment",
    page_icon="🛵",
    layout="wide",
    initial_sidebar_state="expanded",
)


CSS = """
<style>
    :root {
        --bg: #050814;
        --panel: rgba(15, 23, 42, 0.78);
        --panel-strong: rgba(15, 23, 42, 0.94);
        --line: rgba(148, 163, 184, 0.18);
        --ink: #f8fafc;
        --muted: #94a3b8;
        --cyan: #22d3ee;
        --blue: #6366f1;
        --safe: #34d399;
        --caution: #fbbf24;
        --danger: #fb7185;
    }

    .stApp {
        background:
            radial-gradient(circle at 8% 0%, rgba(34, 211, 238, 0.16), transparent 34rem),
            radial-gradient(circle at 90% 4%, rgba(99, 102, 241, 0.18), transparent 30rem),
            linear-gradient(180deg, #060a18 0%, #050814 48%, #020617 100%);
        color: var(--ink);
    }

    .block-container {
        max-width: 1380px;
        padding-top: 1.6rem;
        padding-bottom: 4rem;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background: rgba(2, 6, 23, 0.82);
        border-right: 1px solid var(--line);
    }

    .hero-grid {
        display: grid;
        grid-template-columns: 1.3fr 0.7fr;
        gap: 1.2rem;
        margin-bottom: 1.2rem;
    }

    .hero-card,
    .glass-card,
    .metric-card,
    .verdict-card,
    .frame-card {
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.86), rgba(2, 6, 23, 0.72));
        border: 1px solid var(--line);
        border-radius: 24px;
        box-shadow: 0 24px 70px rgba(0, 0, 0, 0.28);
    }

    .hero-card {
        min-height: 260px;
        overflow: hidden;
        padding: 2rem;
        position: relative;
    }

    .hero-card:after {
        background: radial-gradient(circle, rgba(34, 211, 238, 0.28), transparent 60%);
        content: "";
        height: 320px;
        position: absolute;
        right: -120px;
        top: -120px;
        width: 320px;
    }

    .hero-kicker,
    .section-kicker {
        color: var(--cyan);
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.16em;
        text-transform: uppercase;
    }

    .hero-title {
        color: var(--ink);
        font-size: clamp(2.5rem, 5vw, 5.2rem);
        font-weight: 900;
        letter-spacing: -0.07em;
        line-height: 0.92;
        margin: 0.7rem 0 1rem;
        max-width: 780px;
        position: relative;
        z-index: 1;
    }

    .hero-copy {
        color: #cbd5e1;
        font-size: 1.02rem;
        line-height: 1.75;
        max-width: 720px;
        position: relative;
        z-index: 1;
    }

    .hero-stat {
        padding: 1.35rem;
    }

    .hero-stat-value {
        color: var(--ink);
        font-size: 2.4rem;
        font-weight: 900;
        letter-spacing: -0.05em;
    }

    .hero-stat-label {
        color: var(--muted);
        line-height: 1.55;
        margin-top: 0.4rem;
    }

    .pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 1.2rem;
        position: relative;
        z-index: 1;
    }

    .pill {
        background: rgba(34, 211, 238, 0.08);
        border: 1px solid rgba(34, 211, 238, 0.22);
        border-radius: 999px;
        color: #bae6fd;
        font-size: 0.78rem;
        font-weight: 700;
        padding: 0.42rem 0.7rem;
    }

    .glass-card {
        padding: 1.25rem;
    }

    .section-title {
        color: var(--ink);
        font-size: 1.65rem;
        font-weight: 850;
        letter-spacing: -0.035em;
        margin: 0.15rem 0 1rem;
    }

    .metric-card {
        min-height: 132px;
        padding: 1.15rem;
    }

    .metric-label {
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }

    .metric-value {
        color: var(--ink);
        font-size: 2.05rem;
        font-weight: 900;
        letter-spacing: -0.05em;
        margin-top: 0.4rem;
    }

    .metric-note {
        color: #94a3b8;
        font-size: 0.86rem;
        line-height: 1.45;
        margin-top: 0.3rem;
    }

    .verdict-card {
        border-left: 8px solid var(--cyan);
        margin: 0.8rem 0 1.15rem;
        padding: 1.45rem;
    }

    .verdict-card.safe {
        border-left-color: var(--safe);
        background: linear-gradient(135deg, rgba(52, 211, 153, 0.12), rgba(15, 23, 42, 0.82));
    }

    .verdict-card.moderate {
        border-left-color: var(--caution);
        background: linear-gradient(135deg, rgba(251, 191, 36, 0.12), rgba(15, 23, 42, 0.82));
    }

    .verdict-card.danger {
        border-left-color: var(--danger);
        background: linear-gradient(135deg, rgba(251, 113, 133, 0.13), rgba(15, 23, 42, 0.82));
    }

    .verdict-label {
        color: var(--muted);
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.14em;
        text-transform: uppercase;
    }

    .verdict-value {
        color: var(--ink);
        font-size: clamp(2.2rem, 4vw, 4.2rem);
        font-weight: 950;
        letter-spacing: -0.07em;
        line-height: 1;
        margin: 0.35rem 0;
    }

    .verdict-reason {
        color: #dbeafe;
        font-size: 1rem;
        line-height: 1.6;
        max-width: 900px;
    }

    .frame-card {
        margin: -0.15rem 0 1.1rem;
        padding: 0.95rem;
    }

    .frame-meta {
        color: var(--cyan);
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 0.09em;
        text-transform: uppercase;
    }

    .frame-title {
        color: var(--ink);
        font-size: 1rem;
        font-weight: 800;
        margin-top: 0.25rem;
    }

    .frame-copy {
        color: var(--muted);
        font-size: 0.87rem;
        line-height: 1.48;
        margin-top: 0.25rem;
    }

    div.stButton > button,
    div.stDownloadButton > button {
        background: rgba(15, 23, 42, 0.92) !important;
        border: 1px solid rgba(34, 211, 238, 0.36) !important;
        border-radius: 14px;
        color: #f8fafc !important;
        font-weight: 800;
        min-height: 3rem;
    }

    div.stButton > button:hover,
    div.stDownloadButton > button:hover,
    [data-testid="stFileUploader"] button:hover {
        background: rgba(34, 211, 238, 0.16) !important;
        border-color: rgba(34, 211, 238, 0.72) !important;
        color: #ffffff !important;
    }

    div.stButton > button:focus,
    div.stDownloadButton > button:focus,
    [data-testid="stFileUploader"] button:focus {
        color: #ffffff !important;
        box-shadow: 0 0 0 3px rgba(34, 211, 238, 0.24) !important;
    }

    div.stButton > button:disabled,
    div.stDownloadButton > button:disabled {
        background: rgba(30, 41, 59, 0.7) !important;
        border-color: rgba(148, 163, 184, 0.2) !important;
        color: #94a3b8 !important;
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(120deg, #0891b2, #4f46e5);
        border: 0;
        color: #ffffff !important;
        box-shadow: 0 16px 40px rgba(79, 70, 229, 0.22);
    }

    [data-testid="stFileUploader"] {
        background: var(--panel);
        border: 1px dashed rgba(34, 211, 238, 0.36);
        border-radius: 22px;
        padding: 0.85rem;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: rgba(2, 6, 23, 0.42);
        border: 0;
        border-radius: 16px;
    }

    [data-testid="stFileUploader"] button,
    [data-testid="stFileUploaderDropzone"] button {
        background: rgba(15, 23, 42, 0.95) !important;
        border: 1px solid rgba(34, 211, 238, 0.42) !important;
        border-radius: 12px !important;
        color: #f8fafc !important;
        font-weight: 800 !important;
    }

    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploaderDropzone"] label,
    [data-testid="stFileUploaderDropzone"] small,
    [data-testid="stFileUploaderDropzone"] p {
        color: #cbd5e1 !important;
    }

    hr {
        border-color: var(--line) !important;
    }

    @media (max-width: 900px) {
        .hero-grid {
            grid-template-columns: 1fr;
        }
    }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


LABELS = {0: "Safe", 1: "Caution", 2: "Danger"}


def _display_verdict(verdict):
    return str(verdict).replace("_", " ").title()


def _clean_join(tokens):
    return " / ".join(tokens)


def _readable_description(description):
    return _clean_join(
        token.replace("_", " ").title() for token in str(description).split()
    )


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


def _frame_payload(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    ok, encoded = cv2.imencode(".jpg", frame_bgr)
    return frame_rgb, encoded.tobytes() if ok else b""


def _make_frame_card(result, index, fps, sampling_fps):
    frame_data = result.raw_frame_data[index]
    frame = frame_data.get("frame")
    if frame is None:
        return None
    frame_id = int(frame_data.get("frame_id", index))
    seconds = (frame_id / fps) if fps > 0 else (index / sampling_fps)
    frame_rgb, jpeg_bytes = _frame_payload(frame)
    return {
        "frame": frame_rgb,
        "jpeg_bytes": jpeg_bytes,
        "frame_id": frame_id,
        "time": _format_time(seconds),
        "level": result.numeric_labels[index].title(),
        "description": _readable_description(result.descriptions[index]),
        "score": int(result.numeric_scores[index]),
        "objects": ", ".join(dict.fromkeys(frame_data.get("objects", [])))
        or "No objects detected",
    }


def _build_text_report(data):
    lines = [
        "AUTOMATED TRAFFIC RISK ASSESSMENT REPORT",
        "=" * 72,
        "",
        f"FINAL VERDICT: {_display_verdict(data['verdict'])}",
        f"Reason: {data['verdict_reason']}",
        "",
        "SUMMARY",
        "-" * 72,
        f"Samples analyzed: {data['total_samples']}",
        f"Safe samples: {data['counts']['Safe']}",
        f"Caution samples: {data['counts']['Caution']}",
        f"Danger samples: {data['counts']['Danger']}",
        f"Peak risk score: {data['max_score']}/100",
        f"Average risk score: {data['average_score']:.1f}/100",
        f"Detection failure rate: {data['failure_rate']:.1f}%",
        "",
        "ACTIVE RISK FACTORS",
        "-" * 72,
    ]
    active_factors = [
        (name, count)
        for name, count in data["risk_factors"].items()
        if count > 0
    ]
    if active_factors:
        for name, count in sorted(
            active_factors, key=lambda item: item[1], reverse=True
        ):
            lines.append(f"{name}: {count}")
    else:
        lines.append("(No configured risk factors detected)")

    lines.extend(["", "SAMPLE LOG", "-" * 72])
    for row in data["samples"]:
        lines.append(
            f"{row['Time']} | Frame {row['Frame']} | {row['Level']} | "
            f"Score {row['Score']}/100 | {row['Detected conditions']}"
        )
    return "\n".join(lines) + "\n"


def _build_frames_zip(data):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for category, frames in (
            ("risk_frames", data.get("risk_frames", [])),
            ("safe_frames", data.get("safe_frames", [])),
        ):
            for index, frame in enumerate(frames, 1):
                if not frame.get("jpeg_bytes"):
                    continue
                filename = (
                    f"{category}/{index:02d}_frame_{frame['frame_id']}_"
                    f"{frame['level'].lower()}_score_{frame['score']}.jpg"
                )
                archive.writestr(filename, frame["jpeg_bytes"])
    buffer.seek(0)
    return buffer.getvalue()


def _section(kicker, title, copy=None):
    body = (
        f'<div class="section-kicker">{html.escape(kicker)}</div>'
        f'<div class="section-title">{html.escape(title)}</div>'
    )
    if copy:
        body += f'<div class="metric-note">{html.escape(copy)}</div>'
    st.markdown(body, unsafe_allow_html=True)


def _metric_card(label, value, note):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{html.escape(label)}</div>
            <div class="metric-value">{html.escape(str(value))}</div>
            <div class="metric-note">{html.escape(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _verdict_class(verdict):
    if verdict == "SAFE":
        return "safe"
    if verdict in ("CAUTION", "CAUTION_WITH_DANGER_MOMENT", "MODERATE"):
        return "moderate"
    return "danger"


def _render_frame_grid(frames, empty_message):
    if not frames:
        st.info(empty_message)
        return
    columns = st.columns(2, gap="large")
    for index, evidence in enumerate(frames):
        with columns[index % 2]:
            st.image(evidence["frame"], use_container_width=True)
            st.markdown(
                f"""
                <div class="frame-card">
                    <div class="frame-meta">
                        {html.escape(evidence["level"])} | {html.escape(evidence["time"])}
                        | Frame {evidence["frame_id"]}
                    </div>
                    <div class="frame-title">Risk score {evidence["score"]}/100</div>
                    <div class="frame-copy">
                        {html.escape(evidence["description"])}<br>
                        Objects: {html.escape(evidence["objects"])}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def run_analysis(video_path):
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
                "Level": result.numeric_labels[index].title(),
                "Smoothed level": LABELS.get(int(prediction), "Unknown"),
                "Score": int(score),
                "Speed": str(frame_data.get("ego_speed", "unknown")).title(),
                "Objects": ", ".join(dict.fromkeys(frame_data.get("objects", [])))
                or "None detected",
                "Detected conditions": _readable_description(description),
            }
        )

    risk_indices = [
        index
        for index, label in enumerate(result.numeric_labels)
        if label != "SAFE"
    ]
    risk_indices.sort(
        key=lambda index: (
            result.smoothed_predictions[index],
            result.numeric_scores[index],
        ),
        reverse=True,
    )
    safe_indices = [
        index
        for index, label in enumerate(result.numeric_labels)
        if label == "SAFE"
    ]
    safe_indices.sort(key=lambda index: result.numeric_scores[index])

    risk_frames = [
        card
        for card in (
            _make_frame_card(result, index, fps, config.sampling_fps)
            for index in risk_indices[:8]
        )
        if card is not None
    ]
    safe_frames = [
        card
        for card in (
            _make_frame_card(result, index, fps, config.sampling_fps)
            for index in safe_indices[:8]
        )
        if card is not None
    ]

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
        "risk_frames": risk_frames,
        "safe_frames": safe_frames,
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


def render_results(data):
    sample_frame = pd.DataFrame(data["samples"])
    report_text = _build_text_report(data)
    frames_zip = _build_frames_zip(data)
    metadata = data["metadata"]
    detector_metadata = data["detector_metadata"]

    st.markdown(
        f"""
        <div class="verdict-card {_verdict_class(data["verdict"])}">
            <div class="verdict-label">Final verdict</div>
            <div class="verdict-value">{html.escape(_display_verdict(data["verdict"]))}</div>
            <div class="verdict-reason">{html.escape(data["verdict_reason"])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_columns = st.columns(4)
    with metric_columns[0]:
        _metric_card("Safety rate", f"{data['safe_percentage']:.1f}%", "Safe samples after smoothing")
    with metric_columns[1]:
        _metric_card("Peak risk", f"{data['max_score']}/100", "Highest frame-level score")
    with metric_columns[2]:
        _metric_card("Danger frames", data["counts"]["Danger"], f"{data['danger_percentage']:.1f}% of samples")
    with metric_columns[3]:
        _metric_card("Samples", data["total_samples"], "Frames reviewed by the pipeline")

    if data["incomplete_analysis"]:
        st.warning(
            f"Object detection failed on {data['failure_rate']:.1f}% of samples. "
            "Treat this result as incomplete."
        )

    tabs = st.tabs(["Overview", "Evidence frames", "Sample log", "Downloads"])

    with tabs[0]:
        chart_column, distribution_column = st.columns([1.7, 1], gap="large")
        with chart_column:
            _section("Timeline", "Risk score across the ride")
            st.line_chart(
                pd.DataFrame(data["timeline"]),
                x="Time (seconds)",
                y="Risk score",
                height=330,
                color="#22d3ee",
            )
            st.caption("Score bands: Safe 0-24 | Caution 25-54 | Danger 55-100")

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
            st.bar_chart(distribution, x="Level", y="Samples", height=330, color="#6366f1")

        st.divider()
        left, right = st.columns([1.25, 1], gap="large")
        with left:
            _section("Detected signals", "What contributed to the verdict")
            active_factors = [
                (name, count)
                for name, count in data["risk_factors"].items()
                if count > 0
            ]
            active_factors.sort(key=lambda item: item[1], reverse=True)
            if active_factors:
                for name, count in active_factors[:12]:
                    st.progress(
                        min(count / max(data["total_samples"], 1), 1.0),
                        text=f"{name} - {count} sample{'s' if count != 1 else ''}",
                    )
            else:
                st.success("No configured risk factors were detected.")

        with right:
            _section("Run details", "Video and detector context")
            _metric_card("Average risk", f"{data['average_score']:.1f}/100", "Mean numeric score")
            st.caption(
                f"Video: {_format_time(metadata['duration_seconds'])} | "
                f"{metadata['resolution']} | {metadata['fps']:.1f} FPS"
            )
            st.caption(
                "Detector ensemble: "
                + ", ".join(detector_metadata.get("loaded_models", []))
            )
            st.caption(
                f"Longest danger run: {data['max_danger_run']} | "
                f"Danger episodes: {data['danger_episodes']} | "
                f"Detection failures: {data['failure_rate']:.1f}%"
            )

    with tabs[1]:
        risk_tab, safe_tab = st.tabs(["Risk examples", "Safe examples"])
        with risk_tab:
            _section(
                "Visual evidence",
                "Example risk frames",
                "Caution and danger frames are ordered by severity.",
            )
            _render_frame_grid(
                data["risk_frames"],
                "The pipeline did not classify any analyzed sample as caution or danger.",
            )
        with safe_tab:
            _section(
                "Visual evidence",
                "Example safe frames",
                "Lowest-risk frames from the same upload.",
            )
            _render_frame_grid(
                data["safe_frames"],
                "No safe example frames were available for this run.",
            )

    with tabs[2]:
        _section("Sample log", "Every analyzed observation")
        st.dataframe(
            sample_frame,
            use_container_width=True,
            hide_index=True,
            height=min(620, 48 + (len(sample_frame) * 35)),
            column_config={
                "Score": st.column_config.ProgressColumn(
                    "Score",
                    min_value=0,
                    max_value=100,
                    format="%d",
                ),
            },
        )

    with tabs[3]:
        _section("Export", "Download the analysis package")
        download_columns = st.columns(3)
        download_columns[0].download_button(
            "Download report",
            data=report_text.encode("utf-8"),
            file_name="ride_safety_report.txt",
            mime="text/plain",
            use_container_width=True,
        )
        download_columns[1].download_button(
            "Download sample CSV",
            data=sample_frame.to_csv(index=False).encode("utf-8"),
            file_name="ride_analysis_samples.csv",
            mime="text/csv",
            use_container_width=True,
        )
        download_columns[2].download_button(
            "Download frame ZIP",
            data=frames_zip,
            file_name="ride_example_frames.zip",
            mime="application/zip",
            use_container_width=True,
            disabled=not frames_zip,
        )
        st.caption("The ZIP contains selected safe and risk frames as JPG images.")


def render_header():
    st.markdown(
        """
        <div class="hero-grid">
            <div class="hero-card">
                <div class="hero-kicker">COCO + TTC + optical flow</div>
                <div class="hero-title">Automated Traffic Risk Assessment</div>
                <div class="hero-copy">
                    Upload ride footage and get a motion-aware safety verdict with
                    forward-path object evidence, safe/risk examples, and exportable reports.
                </div>
                <div class="pill-row">
                    <span class="pill">Traffic-jam aware</span>
                    <span class="pill">TTC scoring</span>
                    <span class="pill">COCO object detection</span>
                    <span class="pill">Downloadable outputs</span>
                </div>
            </div>
            <div class="glass-card hero-stat">
                <div class="hero-kicker">How to run</div>
                <div class="hero-stat-value">1 video</div>
                <div class="hero-stat-label">
                    Upload a clip, press analyze, then review the verdict, timeline,
                    evidence frames, and downloadable files.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    st.sidebar.markdown("## Automated Traffic Risk Assessment")
    st.sidebar.caption("Motion-aware motorcycle riding analysis")
    st.sidebar.divider()
    st.sidebar.markdown("### Model setup")
    st.sidebar.caption("Default detector mode: COCO")
    st.sidebar.caption("Models: yolo11n, yolo11m, yolo12m")
    st.sidebar.caption("Risk model: auto-loads models/risk_model.joblib if present")
    st.sidebar.divider()
    st.sidebar.markdown("### Output")
    st.sidebar.caption("Final verdict")
    st.sidebar.caption("Safe and risk example frames")
    st.sidebar.caption("CSV, TXT, and ZIP downloads")


if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "analyzed_upload_id" not in st.session_state:
    st.session_state.analyzed_upload_id = None


render_sidebar()
render_header()

upload_column, preview_column = st.columns([1.05, 0.95], gap="large")
with upload_column:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    _section("Input", "Upload riding footage")
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv", "webm", "m4v"],
        help="Supported formats: MP4, AVI, MOV, MKV, WEBM, and M4V.",
    )
    st.markdown("</div>", unsafe_allow_html=True)

with preview_column:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    _section("Output", "What you will get")
    st.markdown(
        """
        - Final ride verdict and reason
        - Risk timeline and class distribution
        - Example risk frames and safe frames
        - Downloadable report, CSV, and frame ZIP
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None:
    video_bytes = uploaded_file.getvalue()
    upload_id = hashlib.sha256(video_bytes).hexdigest()

    if (
        st.session_state.analyzed_upload_id is not None
        and st.session_state.analyzed_upload_id != upload_id
    ):
        st.session_state.analysis_result = None
        st.session_state.analyzed_upload_id = None

    st.divider()
    video_column, action_column = st.columns([1.55, 0.75], gap="large")
    with video_column:
        st.video(video_bytes)
    with action_column:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        _section("Ready", uploaded_file.name)
        st.caption(f"Size: {len(video_bytes) / (1024 * 1024):.1f} MB")
        analyze_clicked = st.button(
            "Analyze this video",
            type="primary",
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    if analyze_clicked:
        suffix = Path(uploaded_file.name).suffix or ".mp4"
        video_path = None
        status = st.status("Analyzing uploaded footage...", expanded=True)
        try:
            status.write("Preparing temporary video file")
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
                temporary_file.write(video_bytes)
                video_path = temporary_file.name

            status.write("Running detection, tracking, TTC, and risk scoring")
            analysis_data = run_analysis(video_path)
            st.session_state.analysis_result = analysis_data
            st.session_state.analyzed_upload_id = upload_id
            status.update(
                label=f"Analysis complete - {analysis_data['total_samples']} samples reviewed",
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
        st.divider()
        render_results(st.session_state.analysis_result)
else:
    st.divider()
    st.caption("Upload a video to begin. Results are never shown until analysis completes.")
