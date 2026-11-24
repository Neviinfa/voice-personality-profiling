
import io
import os
import html
import warnings

import streamlit as st
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import joblib

from pydub import AudioSegment
import imageio_ffmpeg

warnings.filterwarnings("ignore")

# ==================================================
# CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="VoxSense | AI Emotion Intelligence",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

SAMPLE_RATE = 16000
MODEL_PATH = os.path.join("models", "svm_ravdess_vad_pca.pkl")

# ==================================================
# PREMIUM UI STYLING
# ==================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at top left, #132b45 0%, transparent 45%),
        linear-gradient(135deg, #08111f 0%, #0b1424 55%, #101c30 100%);
    color: #edf4ff;
}

[data-testid="stHeader"] {
    background: rgba(8, 17, 31, 0.85);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101e32, #0a1423);
    border-right: 1px solid rgba(100, 180, 255, 0.15);
}

[data-testid="stSidebar"] * {
    color: #e7f1ff;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}

.hero {
    padding: 35px;
    border-radius: 22px;
    background: linear-gradient(120deg, #142e4c, #102238 60%, #12344b);
    border: 1px solid rgba(70, 190, 255, 0.25);
    box-shadow: 0 15px 45px rgba(0, 0, 0, 0.25);
    margin-bottom: 25px;
}

.hero-label {
    color: #53d7ff;
    font-size: 12px;
    letter-spacing: 3px;
    font-weight: 800;
}

.hero h1 {
    font-size: clamp(28px, 4vw, 43px);
    color: white;
    font-weight: 800;
    margin: 12px 0;
}

.hero p {
    color: #b7cce3;
    font-size: 16px;
    line-height: 1.7;
}

.section-title {
    color: #e9f5ff;
    font-size: 23px;
    font-weight: 750;
    margin: 20px 0 12px 0;
}

.info-card {
    background: linear-gradient(145deg, #162b43, #102033);
    padding: 22px;
    border: 1px solid rgba(89, 190, 255, 0.18);
    border-radius: 17px;
    min-height: 120px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.16);
}

.info-card h3 {
    color: #80dfff;
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 10px;
}

.info-card h2 {
    color: white;
    font-size: 28px;
    font-weight: 800;
    margin: 0;
}

.info-card p {
    color: #9eb5cd;
    font-size: 12px;
}

.result-card {
    background: linear-gradient(135deg, #153e56, #122a43);
    border: 1px solid #35caff;
    border-radius: 20px;
    padding: 30px;
    text-align: center;
    box-shadow: 0 0 35px rgba(31, 190, 255, 0.12);
}

.result-card h1 {
    color: #5cddff;
    font-size: 38px;
    font-weight: 800;
    margin: 10px 0;
}

.result-card p {
    color: #b7d6ec;
    font-size: 13px;
}

.stButton > button {
    background: linear-gradient(90deg, #087eb5, #11b8d9);
    color: white !important;
    border: none;
    border-radius: 12px;
    min-height: 48px;
    font-weight: 750;
    transition: 0.25s ease;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #0ba4dc, #35d6ee);
    box-shadow: 0 5px 22px rgba(0, 190, 255, 0.25);
    transform: translateY(-2px);
}

[data-testid="stFileUploader"] {
    background: rgba(23, 49, 74, 0.65);
    border: 1px dashed #3abfe9;
    border-radius: 16px;
    padding: 15px;
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg, #172d45, #102238);
    padding: 18px;
    border-radius: 15px;
    border: 1px solid rgba(90, 180, 255, 0.18);
}

[data-testid="stMetricLabel"] {
    color: #a8c5df;
}

[data-testid="stMetricValue"] {
    color: #57d7ff;
}

div[data-baseweb="select"] > div {
    background-color: #172b42;
}

hr {
    border-color: rgba(105, 180, 255, 0.17);
}

footer {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# MODEL LOADING
# ==================================================

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    saved = joblib.load(MODEL_PATH)

    return (
        saved["model"],
        list(saved["feature_columns"]),
        saved["classes"]
    )


try:
    model, feature_columns, classes = load_model()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()


# ==================================================
# SPEECH DETECTION
# ==================================================

def detect_speech(audio, sr=SAMPLE_RATE):
    import webrtcvad

    if sr != SAMPLE_RATE:
        audio = librosa.resample(
            audio,
            orig_sr=sr,
            target_sr=SAMPLE_RATE
        )

    audio = np.clip(
        np.asarray(audio, dtype=np.float32),
        -1.0,
        1.0
    )

    vad = webrtcvad.Vad(2)

    pcm = (audio * 32767).astype(np.int16)

    frame_size = int(SAMPLE_RATE * 30 / 1000)
    speech_frames = []

    for start in range(0, len(pcm) - frame_size + 1, frame_size):
        frame = pcm[start:start + frame_size]

        if vad.is_speech(frame.tobytes(), SAMPLE_RATE):
            speech_frames.append(frame)

    if not speech_frames:
        return audio

    result = np.concatenate(speech_frames).astype(np.float32) / 32767.0

    if len(result) < SAMPLE_RATE * 0.25:
        return audio

    return result


# ==================================================
# FEATURE EXTRACTION
# ==================================================

def extract_features(audio, sr=SAMPLE_RATE):
    audio = np.asarray(audio, dtype=np.float32)

    if sr != SAMPLE_RATE:
        audio = librosa.resample(
            audio,
            orig_sr=sr,
            target_sr=SAMPLE_RATE
        )

    if len(audio) < 512:
        raise ValueError("Audio is too short.")

    features = {}

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=13
    )

    for i in range(13):
        features[f"mfcc_{i+1}_mean"] = np.mean(mfcc[i])
        features[f"mfcc_{i+1}_std"] = np.std(mfcc[i])

    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=SAMPLE_RATE
    )

    features["chroma_mean"] = np.mean(chroma)
    features["chroma_std"] = np.std(chroma)

    spectral_features = {
        "spectral_centroid": librosa.feature.spectral_centroid(y=audio, sr=SAMPLE_RATE),
        "spectral_bandwidth": librosa.feature.spectral_bandwidth(y=audio, sr=SAMPLE_RATE),
        "spectral_rolloff": librosa.feature.spectral_rolloff(y=audio, sr=SAMPLE_RATE)
    }

    for name, values in spectral_features.items():
        features[f"{name}_mean"] = np.mean(values)
        features[f"{name}_std"] = np.std(values)

    zcr = librosa.feature.zero_crossing_rate(y=audio)
    features["zero_crossing_rate_mean"] = np.mean(zcr)
    features["zero_crossing_rate_std"] = np.std(zcr)

    rms = librosa.feature.rms(y=audio)
    features["rms_mean"] = np.mean(rms)
    features["rms_std"] = np.std(rms)

    pitch = librosa.yin(
        audio,
        fmin=65,
        fmax=500,
        sr=SAMPLE_RATE
    )

    features["pitch_mean"] = np.mean(pitch)
    features["pitch_std"] = np.std(pitch)

    return features


# ==================================================
# AUDIO LOADER
# ==================================================

def load_audio(uploaded_file):
    extension = uploaded_file.name.lower().split(".")[-1]
    audio_bytes = uploaded_file.getvalue()

    if extension == "wav":
        audio, original_sr = sf.read(io.BytesIO(audio_bytes))

        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)

        audio = audio.astype(np.float32)

        if original_sr != SAMPLE_RATE:
            audio = librosa.resample(
                audio,
                orig_sr=original_sr,
                target_sr=SAMPLE_RATE
            )

    elif extension == "mp3":
        AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()

        segment = AudioSegment.from_file(
            io.BytesIO(audio_bytes),
            format="mp3"
        )

        original_sr = segment.frame_rate

        segment = (
            segment.set_channels(1)
            .set_frame_rate(SAMPLE_RATE)
            .set_sample_width(2)
        )

        audio = np.array(
            segment.get_array_of_samples(),
            dtype=np.float32
        ) / 32768.0

    else:
        raise ValueError("Only WAV and MP3 files are supported.")

    if len(audio) == 0 or not np.isfinite(audio).all():
        raise ValueError("Audio is empty or contains invalid values.")

    return audio, original_sr, extension


# ==================================================
# VISUALIZATION
# ==================================================

def show_waveform(audio):
    fig, ax = plt.subplots(figsize=(12, 3))

    fig.patch.set_facecolor("#13243a")
    ax.set_facecolor("#13243a")

    time_axis = np.arange(len(audio)) / SAMPLE_RATE

    ax.plot(time_axis, audio, color="#48d9ff", linewidth=0.7)

    ax.set_xlabel("Time (seconds)", color="white")
    ax.set_ylabel("Amplitude", color="white")
    ax.tick_params(colors="white")
    ax.grid(alpha=0.15, color="white")

    for spine in ax.spines.values():
        spine.set_color("#34516d")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.markdown(
        """
        <div style="text-align:center;padding:15px 0 22px;">
            <div style="font-size:45px;">🎙️</div>
            <h2 style="color:#53d7ff;margin:5px 0;">VoxSense</h2>
            <p style="color:#9dbbd7;font-size:12px;">
                AI VOICE INTELLIGENCE
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    page = st.radio(
        "MAIN MENU",
        [
            "🏠 Dashboard",
            "🎧 Emotion Detection",
            "📊 Model Analytics",
            "ℹ️ About Project"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown(
        """
        <div style="background:#172e46;padding:15px;border-radius:13px;">
            <p style="color:#58d8ff;font-weight:700;">
                SYSTEM STATUS
            </p>
            <p style="font-size:13px;color:#c2d8ed;">
                🟢 Model loaded<br>
                🟢 Audio processing ready<br>
                🟢 Prediction engine ready
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="text-align:center;color:#7894af;
        font-size:11px;margin-top:35px;">
            VOXSENSE AI<br>
            RAVDESS • PCA • SVM
        </div>
        """,
        unsafe_allow_html=True
    )


# ==================================================
# HERO COMPONENT
# ==================================================

def hero(title, description, label="AI POWERED SPEECH ANALYSIS"):
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-label">{label}</div>
            <h1>{title}</h1>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ==================================================
# DASHBOARD
# ==================================================

if page == "🏠 Dashboard":

    hero(
        "Understand Emotions Beyond Words.",
        "An intelligent speech emotion recognition platform "
        "that analyzes vocal patterns and predicts emotions "
        "using machine learning."
    )

    st.markdown(
        '<div class="section-title">Platform Overview</div>',
        unsafe_allow_html=True
    )

    a, b, c, d = st.columns(4)

    cards = [
        (a, "DATASET", "RAVDESS", "Speech emotion dataset"),
        (b, "AUDIO FILES", "1,440", "Dataset recordings"),
        (c, "FEATURES", "40", "Acoustic features"),
        (d, "TEST ACCURACY", "71.88%", "Recorded experiment")
    ]

    for col, heading, value, caption in cards:
        with col:
            st.markdown(
                f"""
                <div class="info-card">
                    <h3>{heading}</h3>
                    <h2>{value}</h2>
                    <p>{caption}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown(
        '<div class="section-title">Explore the Platform</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns(2)

    with left:
        st.markdown("""
        <div class="info-card">
            <h3>🎧 EMOTION DETECTION</h3>
            <h2>Analyze Voice</h2>
            <p>Upload a WAV or MP3 recording to analyze its vocal features and predict an emotion.</p>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("""
        <div class="info-card">
            <h3>📊 MODEL ANALYTICS</h3>
            <h2>Explore Results</h2>
            <p>Review the recorded training and testing performance of the PCA + SVM model.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("VoxSense AI | Speech Emotion Intelligence Platform")


# ==================================================
# EMOTION DETECTION
# ==================================================

elif page == "🎧 Emotion Detection":

    hero(
        "Emotion Detection Studio",
        "Upload a voice recording and let the AI analyze "
        "its acoustic characteristics."
    )

    st.markdown(
        '<div class="section-title">Upload Your Audio</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=["wav", "mp3"],
        help="Upload WAV or MP3 speech audio."
    )

    if uploaded_file:

        try:
            audio, original_sr, extension = load_audio(uploaded_file)

            st.success(f"{extension.upper()} file uploaded successfully.")

            audio_format = (
                "audio/mpeg" if extension == "mp3" else "audio/wav"
            )

            st.audio(uploaded_file.getvalue(), format=audio_format)

            duration = len(audio) / SAMPLE_RATE

            x, y, z = st.columns(3)

            x.metric("Duration", f"{duration:.2f} sec")
            y.metric("Original Sample Rate", f"{original_sr} Hz")
            z.metric("Processing Rate", "16000 Hz")

            st.markdown(
                '<div class="section-title">Audio Visualization</div>',
                unsafe_allow_html=True
            )

            show_waveform(audio)

            st.markdown("---")

            if st.button(
                "✨ ANALYZE MY VOICE",
                use_container_width=True
            ):

                with st.spinner("Analyzing acoustic features..."):

                    speech = detect_speech(audio)

                    features = extract_features(speech)

                    feature_df = pd.DataFrame([features])
                    feature_df = feature_df.reindex(
                        columns=feature_columns
                    )

                    if feature_df.isnull().any().any():
                        raise ValueError(
                            "Feature mismatch. Check the training "
                            "and prediction feature extraction."
                        )

                    prediction = model.predict(feature_df)[0]

                    probabilities = None
                    score_label = None

                    if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba(
                            feature_df
                        )[0]
                        score_label = "Predicted class probabilities"

                    elif hasattr(model, "decision_function"):
                        scores = np.asarray(
                            model.decision_function(feature_df)
                        )

                        if scores.ndim == 1:
                            scores = np.column_stack([-scores, scores])

                        scores = scores[0] - np.max(scores[0])
                        exp_scores = np.exp(scores)
                        probabilities = exp_scores / np.sum(exp_scores)

                        score_label = (
                            "Relative decision scores — not calibrated probabilities"
                        )

                st.markdown("---")

                st.markdown(
                    '<div class="section-title">AI Analysis Result</div>',
                    unsafe_allow_html=True
                )

                safe_prediction = html.escape(str(prediction))

                st.markdown(
                    f"""
                    <div class="result-card">
                        <p>YOUR VOICE EXPRESSES</p>
                        <h1>{safe_prediction.upper()}</h1>
                        <p>Emotion classification completed</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if probabilities is not None:

                    st.markdown(
                        '<div class="section-title">Emotion Score Analysis</div>',
                        unsafe_allow_html=True
                    )

                    st.caption(score_label)

                    score_df = pd.DataFrame({
                        "Emotion": list(classes),
                        "Score": probabilities
                    }).sort_values(
                        "Score",
                        ascending=False
                    )

                    st.bar_chart(
                        score_df.set_index("Emotion")
                    )

                    for _, row in score_df.iterrows():
                        st.write(
                            f"**{row['Emotion']}** — {row['Score']:.2%}"
                        )

                st.success("Voice analysis completed successfully.")

        except Exception as e:
            st.error(f"Audio processing error: {e}")


# ==================================================
# MODEL ANALYTICS
# ==================================================

elif page == "📊 Model Analytics":

    hero(
        "Model Performance Analytics",
        "An overview of the recorded evaluation results "
        "for the selected PCA + SVM model."
    )

    st.markdown(
        '<div class="section-title">Performance Metrics</div>',
        unsafe_allow_html=True
    )

    a, b = st.columns(2)

    a.metric("Training Accuracy", "99.91%")
    b.metric("Testing Accuracy", "71.88%")

    c, d = st.columns(2)

    c.metric("Balanced Accuracy", "71.02%")
    d.metric("Macro F1 Score", "71.03%")

    st.markdown("---")

    st.subheader("Training vs Testing")

    accuracy = pd.DataFrame({
        "Dataset": ["Training", "Testing"],
        "Accuracy": [99.91, 71.88]
    })

    st.bar_chart(accuracy.set_index("Dataset"))

    st.warning(
        "The difference between training and testing accuracy "
        "indicates possible overfitting. These are recorded "
        "dataset evaluation results, not the confidence or "
        "accuracy of an individual uploaded recording."
    )

    st.markdown("---")

    st.subheader("Model Configuration")

    details = pd.DataFrame({
        "Parameter": [
            "Dataset",
            "Algorithm",
            "Kernel",
            "PCA Components",
            "Original Acoustic Features",
            "SVM C Value"
        ],
        "Configuration": [
            "RAVDESS Speech",
            "Support Vector Machine",
            "RBF",
            "35",
            "40",
            "10"
        ]
    })

    st.dataframe(
        details,
        use_container_width=True,
        hide_index=True
    )


# ==================================================
# ABOUT PROJECT
# ==================================================

elif page == "ℹ️ About Project":

    hero(
        "About VoxSense AI",
        "A machine learning project designed to recognize "
        "human emotions through speech signals."
    )

    st.markdown(
        '<div class="section-title">Project Information</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="info-card">
        <h3>PROJECT OBJECTIVE</h3>
        <p>
        To identify emotional patterns in human speech by extracting
        acoustic characteristics and classifying them using
        machine learning.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Dataset")

    st.write("""
    - RAVDESS Speech Emotion Dataset
    - 24 Actors
    - 1,440 Speech Recordings
    - Eight Emotion Categories
    """)

    st.markdown("### Emotion Categories")

    emotion_cols = st.columns(4)

    emotions = [
        ("😐", "Neutral"),
        ("😌", "Calm"),
        ("😊", "Happy"),
        ("😢", "Sad"),
        ("😠", "Angry"),
        ("😨", "Fearful"),
        ("🤢", "Disgust"),
        ("😲", "Surprised")
    ]

    for i, (emoji, emotion) in enumerate(emotions):
        with emotion_cols[i % 4]:
            st.markdown(
                f"""
                <div class="info-card" style="text-align:center;margin-bottom:12px;">
                    <h2>{emoji}</h2>
                    <h3>{emotion}</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("### Technologies Used")

    st.write("""
    - Python
    - Streamlit
    - Librosa
    - Scikit-learn
    - PCA
    - Support Vector Machine
    - WebRTC VAD
    - PyDub and FFmpeg
    """)

    st.markdown("---")

    st.caption(
        "VoxSense AI | Voice Emotion Recognition | RAVDESS"
    )
