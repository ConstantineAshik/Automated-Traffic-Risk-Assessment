import os
import tempfile
from collections import Counter

import cv2
import pandas as pd
import streamlit as st

from core.config import PipelineConfig
from core.pipeline import analyze


st.set_page_config(
    page_title="Dhaka-Ride Safety Analyzer",
    page_icon="R",
    layout="wide",
)

st.markdown(
    """
<style>
    .reportview-container { background: #0e1117; }
    .main { background: #0e1117; color: white; }
    div.stButton > button:first-child { background-color: #FF4B4B; color: white; }
</style>
""",
    unsafe_allow_html=True,
)


def run_analysis(video_path):
    config = PipelineConfig()
    result = analyze(video_path, config)

    timeline = result.numeric_scores
    danger_frames = []

    for i, pred in enumerate(result.smoothed_predictions):
        if pred != 2 or len(danger_frames) >= 20:
            continue
        data = result.raw_frame_data[i]
        rgb_frame = cv2.cvtColor(data["frame"], cv2.COLOR_BGR2RGB)
        danger_frames.append(
            {
                "frame": rgb_frame,
                "id": data.get("frame_id", i),
                "desc": result.descriptions[i],
                "score": result.numeric_scores[i],
            }
        )

    return {
        "total_frames": result.total_samples,
        "safe_count": result.safe_count,
        "caution_count": result.caution_count,
        "danger_count": result.danger_count,
        "timeline": timeline,
        "danger_frames": danger_frames,
        "risk_factors": Counter(result.stats),
        "verdict": result.verdict,
        "verdict_reason": result.verdict_reason,
    }


st.title("Dhaka-Ride Safety Analyzer")
st.caption("Upload raw riding footage. The system detects risk factors for Dhaka traffic context.")

uploaded_file = st.file_uploader("Upload Video File (MP4, AVI)", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.success("Video uploaded. Ready to analyze.")

    if st.button("Start Safety Audit"):
        with st.spinner("Processing video..."):
            try:
                data = run_analysis(video_path)

                if not data:
                    st.error("Could not process video. It might be corrupt or empty.")
                else:
                    st.divider()

                    kpi1, kpi2, kpi3 = st.columns(3)
                    safety_score = int((data["safe_count"] / data["total_frames"]) * 100)

                    kpi1.metric("Safety Score", f"{safety_score}%", help="Percentage of safe frames")
                    kpi2.metric("Danger Events", data["danger_count"], delta="-Risk", delta_color="inverse")
                    kpi3.metric(
                        "Phone Distraction",
                        data["risk_factors"]["Phone Distraction (5+ frames)"],
                        delta="Critical Risk",
                        delta_color="inverse",
                    )

                    st.subheader("Risk Timeline")
                    chart_data = pd.DataFrame(data["timeline"], columns=["Risk Score"])
                    st.line_chart(chart_data)
                    st.caption("Score 0-30: Safe | 30-65: Caution | 65-100: Danger")

                    st.subheader("Final Verdict")
                    st.write(f"{data['verdict']}: {data['verdict_reason']}")

                    if data["danger_frames"]:
                        st.subheader(f"Critical Events ({len(data['danger_frames'])} samples)")
                        cols = st.columns(2)
                        for idx, frame_obj in enumerate(data["danger_frames"]):
                            with cols[idx % 2]:
                                st.image(frame_obj["frame"], use_column_width=True)
                                st.error(f"Frame {frame_obj['id']} | Score: {frame_obj['score']}")
                                st.code(frame_obj["desc"])
                    else:
                        st.success("No critical danger events detected in the sampled frames.")

            except Exception as exc:
                st.error(f"An error occurred: {exc}")
            finally:
                os.remove(video_path)
