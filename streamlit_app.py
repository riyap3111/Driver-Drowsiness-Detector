from __future__ import annotations

import threading
import time
from collections import deque
import tempfile
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from src.inference import DrowsinessPredictor


st.set_page_config(page_title="Driver Vigilance Intelligence Dashboard", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(255, 209, 102, 0.18), transparent 30%),
            linear-gradient(180deg, #f7f2e7 0%, #efe7da 100%);
    }
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
    }
    .hero {
        padding: 1.4rem 1.6rem;
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(20, 33, 61, 0.96), rgba(61, 90, 128, 0.92));
        color: #fffaf0;
        box-shadow: 0 20px 40px rgba(35, 31, 32, 0.18);
        margin-bottom: 1rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 2.6rem;
        letter-spacing: -0.04em;
    }
    .hero p {
        margin: 0.6rem 0 0 0;
        color: rgba(255, 250, 240, 0.82);
        max-width: 60rem;
    }
    .panel {
        background: rgba(255, 252, 245, 0.8);
        border: 1px solid rgba(20, 33, 61, 0.08);
        border-radius: 20px;
        padding: 1rem 1.1rem;
        box-shadow: 0 10px 25px rgba(27, 38, 59, 0.08);
    }
    .status-chip {
        display: inline-block;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        font-size: 0.88rem;
        font-weight: 600;
        margin-right: 0.45rem;
        margin-bottom: 0.45rem;
    }
    .status-safe {
        background: rgba(46, 125, 50, 0.12);
        color: #1b5e20;
    }
    .status-warn {
        background: rgba(198, 40, 40, 0.12);
        color: #b71c1c;
    }
    .status-neutral {
        background: rgba(20, 33, 61, 0.09);
        color: #1f2937;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>Driver Vigilance Intelligence Dashboard</h1>
        <p>
            A polished safety-monitoring workspace for still images, uploaded road footage, and live webcam scoring.
            The operating threshold is fixed at <strong>75%</strong> to keep decisions conservative for a safety-critical demo.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

available_models = sorted(Path("models").glob("*.pt"))
if not available_models:
    st.warning("Train a model first so the dashboard can load a checkpoint from `models/`.")
    st.code("python -m src.train --data-dir data --model-name efficientnet_b0 --epochs 12 --batch-size 32")
    st.stop()

selected_model = st.sidebar.selectbox("Checkpoint", available_models, format_func=lambda path: path.name)
predictor = DrowsinessPredictor(selected_model)
mode = st.sidebar.radio("Mode", ["Image Review", "Video Review", "Live Monitor"])

st.sidebar.markdown("### Operating Policy")
st.sidebar.metric("Decision threshold", f"{predictor.threshold * 100:.0f}%")
st.sidebar.caption("This threshold is enforced conservatively across image, video, and live monitoring.")
st.sidebar.write("Class labels")
st.sidebar.json(predictor.class_names)


def render_status_badges(result: dict) -> None:
    quality = result.get("quality", {})
    chips = []
    chips.append(
        f"<span class='status-chip {'status-warn' if result['drowsy_score'] >= predictor.threshold else 'status-safe'}'>"
        f"{result['label'].title()} decision"
        "</span>"
    )
    chips.append(
        f"<span class='status-chip {'status-warn' if quality.get('low_light') else 'status-neutral'}'>"
        f"Lighting {'Low' if quality.get('low_light') else 'OK'}"
        "</span>"
    )
    chips.append(
        f"<span class='status-chip {'status-warn' if quality.get('blurry') else 'status-neutral'}'>"
        f"Sharpness {'Low' if quality.get('blurry') else 'OK'}"
        "</span>"
    )
    st.markdown("".join(chips), unsafe_allow_html=True)


if mode == "Image Review":
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a driver face image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file).convert("RGB")
        bgr_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        result = predictor.predict(bgr_image, with_explanation=True)

        left, right = st.columns([1.15, 1])
        with left:
            st.image(pil_image, caption="Input image", use_container_width=True)
            st.image(result["attention_heatmap"], caption="Model attention map", use_container_width=True)
        with right:
            render_status_badges(result)
            st.metric("Drowsiness risk", f"{result['drowsy_score'] * 100:.2f}%")
            st.metric("Model confidence", f"{result['confidence'] * 100:.2f}%")
            st.metric("Eyes detected", result["eye_count"])
            st.metric("Brightness score", f"{result['quality']['brightness']:.1f}")
            st.metric("Blur score", f"{result['quality']['blur_score']:.1f}")
            st.json({key: round(value, 4) for key, value in result["scores"].items()})
            if result["drowsy_score"] >= predictor.threshold:
                st.error("Risk is above the 75% safety threshold.")
            else:
                st.success("Risk remains below the 75% safety threshold.")

        tabs = st.tabs(["Face Crop", "Quality Notes"])
        with tabs[0]:
            st.image(result["face_crop_rgb"], use_container_width=False)
        with tabs[1]:
            notes = []
            if result["quality"]["low_light"]:
                notes.append("Lighting is low. Front lighting will usually improve reliability.")
            if result["quality"]["blurry"]:
                notes.append("Frame sharpness is weak. Reduce camera motion or improve focus.")
            if result["eye_count"] == 0:
                notes.append("No eyes were detected in the crop. A more frontal face angle may help.")
            if not notes:
                notes.append("Capture quality looks acceptable for inference.")
            for note in notes:
                st.write(f"- {note}")
    st.markdown("</div>", unsafe_allow_html=True)

elif mode == "Video Review":
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    uploaded_video = st.file_uploader("Upload a driving video", type=["mp4", "mov", "avi", "mkv"])
    sample_rate = st.sidebar.slider("Analyze every Nth frame", min_value=1, max_value=15, value=4)
    smoothing_window = st.sidebar.slider("Video smoothing window", min_value=4, max_value=30, value=12)
    if uploaded_video is not None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = Path(tmp_dir) / uploaded_video.name
            output_video_path = Path(tmp_dir) / f"annotated_{uploaded_video.name}.mp4"
            video_path.write_bytes(uploaded_video.read())

            with st.spinner("Analyzing video frames and computing vigilance timeline..."):
                result = predictor.analyze_video(
                    video_path,
                    sample_every_n_frames=sample_rate,
                    smoothing_window=smoothing_window,
                    output_video_path=output_video_path,
                )

            timeline_df = pd.DataFrame(result["timeline"])
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Max drowsiness score", f"{result['max_drowsy_score'] * 100:.2f}%")
            k2.metric("Mean smoothed score", f"{result['mean_drowsy_score'] * 100:.2f}%")
            k3.metric("Drowsy frame ratio", f"{result['drowsy_ratio'] * 100:.2f}%")
            high_risk_events = int((timeline_df["smoothed_drowsy_score"] >= predictor.threshold).sum()) if not timeline_df.empty else 0
            k4.metric("High-risk moments", high_risk_events)

            if not timeline_df.empty:
                timeline_df["Risk Band"] = np.where(
                    timeline_df["smoothed_drowsy_score"] >= predictor.threshold,
                    "High",
                    np.where(timeline_df["smoothed_drowsy_score"] >= 0.5, "Moderate", "Low"),
                )
                st.line_chart(
                    timeline_df.set_index("timestamp_sec")[["drowsy_score", "smoothed_drowsy_score"]],
                    use_container_width=True,
                )
                st.dataframe(
                    timeline_df[["timestamp_sec", "label", "drowsy_score", "smoothed_drowsy_score", "Risk Band"]].tail(25),
                    use_container_width=True,
                )

            if output_video_path.exists():
                st.subheader("Annotated Video")
                st.video(output_video_path.read_bytes())
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.subheader("Live Webcam Risk Monitor")
    st.caption("Click Start, allow camera access, and monitor a smoother live drowsiness signal with camera-quality checks.")

    smoothing_window = st.sidebar.slider("Live smoothing window", min_value=5, max_value=40, value=16)
    frame_skip = st.sidebar.slider("Analyze every Nth live frame", min_value=1, max_value=6, value=2)
    ema_alpha = st.sidebar.slider("Live score smoothing", min_value=0.05, max_value=0.5, value=0.18, step=0.01)
    high_risk_patience = st.sidebar.slider("High-risk alert patience", min_value=2, max_value=20, value=6)

    lock = threading.Lock()
    live_state = {
        "frame_index": 0,
        "result": None,
        "timeline": deque(maxlen=240),
        "ema_score": None,
        "consecutive_high_risk": 0,
        "alerts_triggered": 0,
    }

    def live_video_frame_callback(frame):
        image = frame.to_ndarray(format="bgr24")
        live_state["frame_index"] += 1

        if live_state["frame_index"] % frame_skip != 0:
            return av.VideoFrame.from_ndarray(image, format="bgr24")

        result = predictor.predict(image, with_explanation=False)
        with lock:
            previous_ema = live_state["ema_score"]
            ema_score = result["drowsy_score"] if previous_ema is None else (ema_alpha * result["drowsy_score"] + (1 - ema_alpha) * previous_ema)
            history_scores = [item["ema_score"] for item in live_state["timeline"]][- (smoothing_window - 1) :]
            smoothed_score = float(np.mean(history_scores + [ema_score]))
            high_risk = smoothed_score >= predictor.threshold
            live_state["consecutive_high_risk"] = live_state["consecutive_high_risk"] + 1 if high_risk else 0
            if live_state["consecutive_high_risk"] == high_risk_patience:
                live_state["alerts_triggered"] += 1

            live_state["ema_score"] = ema_score
            live_result = {
                "timestamp": time.time(),
                "label": "sleepy" if high_risk else "awake",
                "drowsy_score": result["drowsy_score"],
                "ema_score": ema_score,
                "smoothed_drowsy_score": smoothed_score,
                "eye_count": result["eye_count"],
                "confidence": result["confidence"],
                "face_bbox": result["face_bbox"],
                "quality": result["quality"],
                "risk_level": "high" if high_risk else "moderate" if smoothed_score >= 0.5 else "low",
                "consecutive_high_risk": live_state["consecutive_high_risk"],
                "alerts_triggered": live_state["alerts_triggered"],
            }
            live_state["result"] = live_result
            live_state["timeline"].append(live_result)

        annotated = image.copy()
        bbox = result["face_bbox"]
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (245, 158, 11), 2)

        quality = result["quality"]
        color = (0, 190, 90) if smoothed_score < 0.5 else (0, 165, 255) if smoothed_score < predictor.threshold else (0, 0, 255)
        label = "ALERT" if smoothed_score < 0.5 else "WATCH" if smoothed_score < predictor.threshold else "DROWSY"
        cv2.putText(annotated, f"{label} risk={smoothed_score:.2f}", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(annotated, f"raw={result['drowsy_score']:.2f} eyes={result['eye_count']} conf={result['confidence']:.2f}", (20, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        cv2.putText(annotated, f"light={quality['brightness']:.0f} blur={quality['blur_score']:.0f}", (20, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    ctx = webrtc_streamer(
        key="live-drowsiness-monitor",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": {"width": {"ideal": 960}, "height": {"ideal": 540}}, "audio": False},
        video_frame_callback=live_video_frame_callback,
        async_processing=True,
    )

    top_metrics = st.columns(5)
    chart_placeholder = st.empty()
    table_placeholder = st.empty()
    alert_placeholder = st.empty()

    if ctx.state.playing:
        while ctx.state.playing:
            with lock:
                current = dict(live_state["result"]) if live_state["result"] is not None else None
                timeline = list(live_state["timeline"])

            if current is not None:
                top_metrics[0].metric("Instant risk", f"{current['drowsy_score'] * 100:.1f}%")
                top_metrics[1].metric("Smoothed risk", f"{current['smoothed_drowsy_score'] * 100:.1f}%")
                top_metrics[2].metric("Eyes detected", current["eye_count"])
                top_metrics[3].metric("Alerts triggered", current["alerts_triggered"])
                top_metrics[4].metric("State", current["label"].title())

                if current["consecutive_high_risk"] >= high_risk_patience:
                    alert_placeholder.error("Sustained high-risk drowsiness detected. This session would trigger an intervention.")
                elif current["quality"]["low_light"] or current["quality"]["blurry"]:
                    alert_placeholder.warning("Camera quality is limiting reliability. Improve lighting or reduce motion.")
                else:
                    alert_placeholder.success("Feed quality is acceptable and the live score is stable.")

                timeline_df = pd.DataFrame(timeline)
                if not timeline_df.empty:
                    timeline_df["step"] = np.arange(len(timeline_df))
                    chart_placeholder.line_chart(
                        timeline_df.set_index("step")[["drowsy_score", "ema_score", "smoothed_drowsy_score"]],
                        use_container_width=True,
                    )
                    table_placeholder.dataframe(
                        timeline_df[
                            [
                                "label",
                                "risk_level",
                                "drowsy_score",
                                "ema_score",
                                "smoothed_drowsy_score",
                                "eye_count",
                                "consecutive_high_risk",
                            ]
                        ].tail(12),
                        use_container_width=True,
                    )
            time.sleep(0.2)
    else:
        st.info("Press Start above to begin live scoring.")
    st.markdown("</div>", unsafe_allow_html=True)
