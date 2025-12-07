import streamlit as st
import cv2
import tempfile
import os
import shutil
from collections import deque, Counter
import pandas as pd

# Import your modules
from video_processor import VideoProcessor
from text_generator import TextGenerator
from risk_model import RiskModel
from risk_calculator import RiskCalculator

# --- Page Config ---
st.set_page_config(
    page_title="Dhaka-Ride Safety Analyzer",
    page_icon="🏍️",
    layout="wide"
)

# --- CSS for Styling ---
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .main {
        background: #0e1117;
        color: white;
    }
    div.stButton > button:first-child {
        background-color: #FF4B4B;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Function: The Analysis Pipeline ---
def run_analysis(video_path):
    # 1. Initialize Components
    # We use a temp directory for debug images if needed, but mostly we use memory
    temp_dir = tempfile.mkdtemp()
    processor = VideoProcessor(video_path, window_size=10, danger_img_dir=temp_dir)
    text_gen = TextGenerator()
    risk_model = RiskModel()
    risk_calc = RiskCalculator()

    # Train the mock model immediately
    risk_model.train_mock_model()

    # 2. Process Video
    # In a web app, we might limit frames to prevent timeouts on free tier
    # But here we run the full processor you wrote
    raw_frame_data = processor.process_video()

    if not raw_frame_data:
        return None

    # 3. Generate Descriptions & Predict
    descriptions = []
    for frame_data in raw_frame_data:
        desc = text_gen.generate_description(frame_data)
        descriptions.append(desc)

    # 4. Compute numeric scores first
    numeric_scores = []
    for frame_data in raw_frame_data:
        ego_speed = frame_data.get("ego_speed", "slow")
        if ego_speed == "fast": speed_cat = "moderate"
        elif ego_speed == "stationary": speed_cat = "stationary"
        else: speed_cat = "slow"
        score = risk_calc.calculate_risk_score(frame_data, speed_cat)
        numeric_scores.append(score)

    # 5. ML Prediction
    model_preds = risk_model.predict_risk(descriptions)

    # 6. Merge numeric label and model label (numeric trusted on extremes)
    def merge_labels(model_label, numeric_score):
        numeric_label = risk_calc.score_to_label(numeric_score)
        if numeric_label == "DANGER":
            return 2
        if numeric_label == "SAFE" and int(model_label) == 0:
            return 0
        if int(model_label) == 2 or numeric_label == "CAUTION":
            return 1
        return int(model_label)

    merged_preds = [merge_labels(m, s) for m, s in zip(model_preds, numeric_scores)]

    # 7. Temporal smoothing on merged labels (keep max in window)
    window_size = 5
    risk_window = deque(maxlen=window_size)
    smoothed_predictions = []
    for lab in merged_preds:
        risk_window.append(int(lab))
        smoothed_predictions.append(max(risk_window))

    # 8. Package Results
    results = {
        "total_frames": len(raw_frame_data),
        "safe_count": smoothed_predictions.count(0),
        "caution_count": smoothed_predictions.count(1),
        "danger_count": smoothed_predictions.count(2),
        "timeline": numeric_scores,
        "danger_frames": [],
        "risk_factors": Counter()
    }

    # Extract Danger Frames & Stats
    for i, (pred, data, desc) in enumerate(zip(smoothed_predictions, raw_frame_data, descriptions)):
        # Count risk keywords for the stats board
        if "phone_distraction" in desc: results["risk_factors"]["Phone Use"] += 1
        if "wrong_side" in desc: results["risk_factors"]["Wrong Side"] += 1
        if "high_speed" in desc: results["risk_factors"]["High Speed"] += 1
        if "pedestrian" in desc: results["risk_factors"]["Pedestrian"] += 1
        
        # Save Danger Frames for Gallery (Limit to 20 to save memory)
        if pred == 2 and len(results["danger_frames"]) < 20:
            # Convert BGR to RGB for Streamlit
            rgb_frame = cv2.cvtColor(data["frame"], cv2.COLOR_BGR2RGB)
            results["danger_frames"].append({
                "frame": rgb_frame,
                "id": data["frame_id"],
                "desc": desc,
                "score": numeric_scores[i]
            })

    # Cleanup temp dir
    shutil.rmtree(temp_dir)
    return results

# --- Main UI ---
st.title("🏍️ Dhaka-Ride Safety Analyzer V9")
st.caption("Upload raw riding footage. AI detects risk factors specific to Dhaka traffic context.")

uploaded_file = st.file_uploader("Upload Video File (MP4, AVI)", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save uploaded file to a temporary path because OpenCV needs a real file path
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.success("Video uploaded! Ready to analyze.")
    
    if st.button("🚀 Start Safety Audit"):
        with st.spinner("Processing video... (This may take a minute)"):
            try:
                data = run_analysis(video_path)
                
                if not data:
                    st.error("Could not process video. It might be corrupt or empty.")
                else:
                    # --- DASHBOARD ---
                    st.divider()
                    
                    # 1. Top Level Metrics
                    kpi1, kpi2, kpi3 = st.columns(3)
                    safety_score = int((data['safe_count'] / data['total_frames']) * 100)
                    
                    kpi1.metric("Safety Score", f"{safety_score}%", help="Percentage of safe frames")
                    kpi2.metric("Danger Events", data['danger_count'], delta="-Risk", delta_color="inverse")
                    kpi3.metric("Phone Distraction", data['risk_factors']['Phone Use'], delta="Critical Risk", delta_color="inverse")

                    # 2. Risk Timeline
                    st.subheader("📈 Risk Timeline")
                    chart_data = pd.DataFrame(data['timeline'], columns=["Risk Score"])
                    st.line_chart(chart_data)
                    st.caption("Score 0-30: Safe | 30-65: Caution | 65-100: Danger")

                    # 3. Danger Evidence Gallery
                    if data['danger_frames']:
                        st.subheader(f"🚨 Critical Events ({len(data['danger_frames'])} samples)")
                        
                        cols = st.columns(2)
                        for idx, frame_obj in enumerate(data['danger_frames']):
                            with cols[idx % 2]:
                                st.image(frame_obj['frame'], use_column_width=True)
                                st.error(f"Frame {frame_obj['id']} | Score: {frame_obj['score']}")
                                st.code(frame_obj['desc'])
                    else:
                        st.balloons()
                        st.success("No critical danger events detected in the sampled frames!")

            except Exception as e:
                st.error(f"An error occurred: {e}")
            finally:
                os.remove(video_path)
