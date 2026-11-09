import os
import tempfile

import streamlit as st
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
import joblib
import webrtcvad

from transformers import Wav2Vec2Processor, Wav2Vec2Model
from src.enhanced_feature_extraction import extract_features


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="VoiceAI | Emotion Intelligence",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at 10% 0%, rgba(91, 55, 190, 0.20), transparent 38%),
        radial-gradient(ellipse at 90% 10%, rgba(20, 130, 190, 0.15), transparent 35%),
        #080B16;
    color: #F4F5FF;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

section[data-testid="stSidebar"] {
    background: #0D1120;
    border-right: 1px solid rgba(155, 140, 255, 0.18);
}

section[data-testid="stSidebar"] * {
    color: #E7E9FF;
}

.hero {
    padding: 34px;
    border-radius: 24px;
    background: linear-gradient(120deg, rgba(105, 65, 220, .27),
        rgba(27, 119, 190, .16), rgba(16, 22, 42, .92));
    border: 1px solid rgba(164, 145, 255, .28);
    box-shadow: 0 15px 50px rgba(0,0,0,.25);
    margin-bottom: 25px;
}

.eyebrow {
    color: #B9A9FF;
    font-size: 12px;
    letter-spacing: 3px;
    font-weight: 800;
    text-transform: uppercase;
}

.hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(32px, 4vw, 52px);
    font-weight: 800;
    line-height: 1.1;
    margin: 12px 0;
    background: linear-gradient(90deg, #FFFFFF, #B9A5FF, #75D9FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: #C1C7E2;
    font-size: 15px;
    max-width: 750px;
    line-height: 1.7;
}

.section-heading {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 23px;
    font-weight: 700;
    color: #F3F1FF;
    margin: 22px 0 12px 0;
}

.metric-card {
    background: linear-gradient(145deg, rgba(28, 34, 61, .96),
        rgba(16, 21, 40, .96));
    border: 1px solid rgba(150, 135, 255, .20);
    border-radius: 18px;
    padding: 21px;
    min-height: 125px;
    box-shadow: 0 8px 25px rgba(0,0,0,.16);
}

.metric-label {
    color: #9DA8CC;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 700;
}

.metric-value {
    color: #FFFFFF;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 30px;
    font-weight: 800;
    margin-top: 10px;
}

.metric-note {
    color: #8E9ABF;
    font-size: 12px;
    margin-top: 4px;
}

.panel {
    background: rgba(18, 24, 45, .88);
    border: 1px solid rgba(145, 133, 255, .20);
    border-radius: 20px;
    padding: 23px;
    margin-bottom: 16px;
}

.result-panel {
    background: linear-gradient(135deg, rgba(105, 65, 220, .25),
        rgba(22, 105, 165, .18), rgba(18, 24, 45, .95));
    border: 1px solid rgba(165, 143, 255, .38);
    border-radius: 22px;
    padding: 30px;
    text-align: center;
    margin: 15px 0;
}

.result-label {
    color: #B9ACFF;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-size: 12px;
    font-weight: 800;
}

.result-value {
    color: #FFFFFF;
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(35px, 5vw, 55px);
    font-weight: 800;
    margin: 12px 0;
}

.upload-hint {
    color: #9FAAD0;
    font-size: 13px;
    text-align: center;
    padding: 8px;
}

.emotion-chip {
    background: rgba(115, 92, 220, .15);
    border: 1px solid rgba(157, 137, 255, .25);
    padding: 13px;
    border-radius: 13px;
    text-align: center;
    color: #E6E1FF;
    font-weight: 700;
}

.footer {
    color: #7581A5;
    text-align: center;
    font-size: 12px;
    border-top: 1px solid rgba(145, 133, 255, .15);
    padding-top: 20px;
    margin-top: 35px;
}

div.stButton > button {
    background: linear-gradient(90deg, #7955E8, #287CCB);
    color: white;
    border: 0;
    border-radius: 12px;
    padding: 12px 25px;
    font-weight: 800;
    min-height: 48px;
    box-shadow: 0 6px 22px rgba(105, 80, 220, .25);
}

div.stButton > button:hover {
    border: 1px solid #B9A5FF;
    color: white;
    filter: brightness(1.12);
}

[data-testid="stFileUploader"] {
    background: rgba(32, 39, 69, .55);
    border: 1px dashed rgba(165, 145, 255, .55);
    border-radius: 17px;
    padding: 15px;
}

.stAlert {
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# PROJECT SETTINGS
# =========================================================

MODEL_PATH = "models/svm_combined.pkl"
FEATURE_PATH = "features/enhanced_train.csv"

SAMPLE_RATE = 16000
FRAME_DURATION_MS = 30
VAD_MODE = 2

EMOTION_INFO = {
    "anger": "The model identified speech patterns associated with anger.",
    "disgust": "The model identified speech patterns associated with disgust.",
    "fear": "The model identified speech patterns associated with fear.",
    "happy": "The model identified speech patterns associated with happiness.",
    "neutral": "The model identified speech patterns associated with a neutral emotion.",
    "sad": "The model identified speech patterns associated with sadness."
}


# =========================================================
# LOAD MODEL AND FEATURE EXTRACTOR
# =========================================================

@st.cache_resource
def load_svm():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_wav2vec():
    model_name = "facebook/wav2vec2-base"

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)
    model.eval()

    return processor, model


@st.cache_data
def load_feature_columns():
    df = pd.read_csv(FEATURE_PATH, nrows=1)

    metadata = ["filename", "emotion", "speaker_id"]

    return [
        column for column in df.columns
        if column not in metadata
    ]


# =========================================================
# VAD PREPROCESSING
# =========================================================

def apply_vad(audio):

    audio_int16 = np.int16(
        np.clip(audio, -1, 1) * 32767
    )

    vad = webrtcvad.Vad(VAD_MODE)

    frame_size = int(
        SAMPLE_RATE * FRAME_DURATION_MS / 1000
    )

    speech_frames = []

    for start in range(
        0,
        len(audio_int16) - frame_size + 1,
        frame_size
    ):

        frame = audio_int16[
            start:start + frame_size
        ]

        if vad.is_speech(frame.tobytes(), SAMPLE_RATE):
            speech_frames.append(frame)

    if not speech_frames:
        return audio.astype(np.float32)

    clean_audio = np.concatenate(speech_frames)

    return clean_audio.astype(np.float32) / 32767.0


# =========================================================
# WAV2VEC2 EMBEDDING
# =========================================================

def extract_embedding(audio, processor, model):

    inputs = processor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt"
    )

    with torch.inference_mode():
        outputs = model(**inputs)

    embedding = (
        outputs.last_hidden_state
        .mean(dim=1)
        .squeeze()
        .cpu()
        .numpy()
    )

    return embedding.astype(np.float32)


# =========================================================
# AUDIO PREDICTION
# =========================================================

def predict_emotion(audio_path):

    svm = load_svm()
    processor, wav2vec_model = load_wav2vec()
    acoustic_columns = load_feature_columns()

    if len(acoustic_columns) != 343:
        raise ValueError(
            f"Expected 343 acoustic features; found {len(acoustic_columns)}."
        )

    expected_features = svm.named_steps["scaler"].n_features_in_

    if expected_features != 1111:
        raise ValueError(
            f"Model expects {expected_features} features instead of 1111."
        )

    # Load audio using the same librosa approach as training
    audio, _ = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    audio = np.nan_to_num(
        audio.astype(np.float32)
    )

    if len(audio) == 0:
        raise ValueError("The uploaded audio is empty.")

    clean_audio = apply_vad(audio)

    # Save cleaned recording temporarily
    temp_clean_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as temp_file:
            temp_clean_path = temp_file.name

        sf.write(
            temp_clean_path,
            clean_audio,
            SAMPLE_RATE
        )

        acoustic_features = extract_features(
            temp_clean_path
        )

        if acoustic_features is None:
            raise ValueError("Acoustic feature extraction failed.")

        acoustic_vector = np.array(
            [
                acoustic_features[column]
                for column in acoustic_columns
            ],
            dtype=np.float32
        )

    finally:
        if (
            temp_clean_path
            and os.path.exists(temp_clean_path)
        ):
            os.remove(temp_clean_path)

    embedding = extract_embedding(
        clean_audio,
        processor,
        wav2vec_model
    )

    if len(embedding) != 768:
        raise ValueError(
            f"Expected 768 embedding values; found {len(embedding)}."
        )

    combined_vector = np.concatenate(
        [acoustic_vector, embedding]
    )

    if len(combined_vector) != 1111:
        raise ValueError("Combined feature count is not 1111.")

    X = combined_vector.reshape(1, -1)

    return svm.predict(X)[0]


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        "<h1 style='font-family:Space Grotesk;color:#B9A5FF;'>◈ VOICE<span style='color:#75D9FF;'>AI</span></h1>",
        unsafe_allow_html=True
    )

    st.caption("EMOTION INTELLIGENCE PLATFORM")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "🏠  Dashboard",
            "🎙️  Voice Analysis",
            "🧠  About the Model",
            "📊  Project Overview"
        ],
        label_visibility="visible"
    )

    st.divider()

    st.markdown("**SYSTEM STATUS**")

    st.markdown(
        "<span style='color:#55DDAA;'>●</span> Application interface ready",
        unsafe_allow_html=True
    )

    st.caption("Combined SVM research prototype")

    st.divider()

    st.caption("CREMA-D emotion recognition project")


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="hero">
    <div class="eyebrow">ARTIFICIAL INTELLIGENCE • SPEECH ANALYTICS</div>
    <h1>Voice Intelligence<br>Redefined.</h1>
    <p>
        Explore emotion recognition through acoustic intelligence
        and deep speech representations. Upload a voice recording
        and let the combined machine learning model analyze it.
    </p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# DASHBOARD PAGE
# =========================================================

if page == "🏠  Dashboard":

    st.markdown(
        '<div class="section-heading">Platform Overview</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    metrics = [
        (c1, "MODEL ACCURACY", "61.12%", "Experimental test result"),
        (c2, "ACOUSTIC FEATURES", "343", "Handcrafted features"),
        (c3, "DEEP EMBEDDINGS", "768", "Wav2Vec2 dimensions"),
        (c4, "TOTAL FEATURES", "1,111", "Combined feature vector")
    ]

    for column, label, value, note in metrics:
        with column:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-note">{note}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-heading">Emotion Recognition</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="panel">
        <h3 style="color:#C4B5FF;">Understand the voice beyond words.</h3>
        <p style="color:#AAB4D2;line-height:1.8;">
            This research prototype combines traditional acoustic
            features with Wav2Vec2 deep representations to classify
            six emotion categories from speech recordings.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Supported Emotion Categories")

    emotion_cols = st.columns(6)

    emotions = [
        ("😠", "ANGER"),
        ("🤢", "DISGUST"),
        ("😨", "FEAR"),
        ("😊", "HAPPY"),
        ("😐", "NEUTRAL"),
        ("😢", "SAD")
    ]

    for col, (emoji, emotion) in zip(emotion_cols, emotions):
        with col:
            st.markdown(
                f'<div class="emotion-chip">{emoji}<br>{emotion}</div>',
                unsafe_allow_html=True
            )

    st.markdown(
        '<div class="section-heading">How the system works</div>',
        unsafe_allow_html=True
    )

    steps = st.columns(4)

    workflow = [
        ("01", "Audio Input", "Upload a voice recording."),
        ("02", "Preprocessing", "Convert to mono 16 kHz and apply VAD."),
        ("03", "Feature Extraction", "Extract acoustic and Wav2Vec2 features."),
        ("04", "Prediction", "Combined SVM predicts an emotion.")
    ]

    for col, (number, title, description) in zip(steps, workflow):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="eyebrow">{number}</div>
                <h4 style="color:#FFFFFF;">{title}</h4>
                <p style="font-size:13px;color:#AAB4D2;">{description}</p>
            </div>
            """, unsafe_allow_html=True)


# =========================================================
# VOICE ANALYSIS PAGE
# =========================================================

elif page == "🎙️  Voice Analysis":

    st.markdown(
        '<div class="section-heading">Voice Analysis Studio</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="panel">
        <h3 style="color:#C4B5FF;">Upload your voice recording</h3>
        <p style="color:#AAB4D2;">
            Supported formats: WAV and MP3. Audio is converted to
            mono 16 kHz before feature extraction.
        </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose your audio recording",
        type=["wav", "mp3"],
        help="Upload a WAV or MP3 voice recording."
    )

    if uploaded_file is not None:

        audio_bytes = uploaded_file.getvalue()

        st.markdown(
            '<div class="section-heading">Recording Preview</div>',
            unsafe_allow_html=True
        )

        st.audio(
            audio_bytes,
            format=uploaded_file.type
        )

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">SELECTED RECORDING</div>
            <div style="color:white;font-size:19px;font-weight:700;margin-top:9px;">
                🎵 {uploaded_file.name}
            </div>
            <div class="metric-note">
                File size: {len(audio_bytes) / (1024 * 1024):.2f} MB
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button(
            "✦  ANALYZE VOICE",
            type="primary",
            use_container_width=True
        ):

            temp_upload_path = None

            try:
                suffix = os.path.splitext(uploaded_file.name)[1].lower()

                with tempfile.NamedTemporaryFile(
                    suffix=suffix,
                    delete=False
                ) as temp_file:

                    temp_upload_path = temp_file.name
                    temp_file.write(audio_bytes)

                with st.spinner(
                    "Processing audio • Extracting features • Running AI prediction..."
                ):

                    prediction = predict_emotion(
                        temp_upload_path
                    )

                st.markdown(
                    '<div class="section-heading">Analysis Result</div>',
                    unsafe_allow_html=True
                )

                emotion_name = str(prediction).strip().lower()

                emoji_map = {
                    "anger": "😠",
                    "disgust": "🤢",
                    "fear": "😨",
                    "happy": "😊",
                    "neutral": "😐",
                    "sad": "😢"
                }

                emoji = emoji_map.get(emotion_name, "🎯")

                st.markdown(f"""
                <div class="result-panel">
                    <div class="result-label">PREDICTED EMOTION</div>
                    <div style="font-size:48px;margin:10px;">{emoji}</div>
                    <div class="result-value">{emotion_name.upper()}</div>
                    <p style="color:#B8C2E0;">
                        {EMOTION_INFO.get(
                            emotion_name,
                            "The model returned an emotion label."
                        )}
                    </p>
                </div>
                """, unsafe_allow_html=True)

                st.success("Voice analysis completed successfully.")

                st.caption(
                    "This is a model prediction, not a calibrated confidence score."
                )

            except Exception as error:

                st.error(f"Prediction failed: {error}")

                st.info(
                    "Check the terminal for the full error details."
                )

            finally:
                if (
                    temp_upload_path
                    and os.path.exists(temp_upload_path)
                ):
                    os.remove(temp_upload_path)

    else:

        st.markdown("""
        <div class="panel" style="text-align:center;padding:35px;">
            <div style="font-size:48px;">🎙️</div>
            <h3 style="color:#C4B5FF;">Your voice analysis starts here</h3>
            <p style="color:#AAB4D2;">
                Upload a recording above to begin the analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# ABOUT THE MODEL PAGE
# =========================================================

elif page == "🧠  About the Model":

    st.markdown(
        '<div class="section-heading">Inside the AI Model</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="panel">
        <h3 style="color:#C4B5FF;">Combined SVM Architecture</h3>
        <p style="color:#AAB4D2;line-height:1.8;">
            The model combines handcrafted acoustic information
            and Wav2Vec2 deep speech representations into one
            feature vector. A StandardScaler and RBF-kernel
            Support Vector Machine perform the final classification.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Feature Architecture")

    left, right = st.columns(2)

    with left:
        st.markdown("""
        <div class="metric-card">
            <div class="eyebrow">HANDCRAFTED FEATURES</div>
            <div class="metric-value">343</div>
            <p style="color:#AAB4D2;">
                Acoustic and prosodic information such as MFCC,
                energy, pitch, spectral characteristics and more.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("""
        <div class="metric-card">
            <div class="eyebrow">WAV2VEC2 EMBEDDINGS</div>
            <div class="metric-value">768</div>
            <p style="color:#AAB4D2;">
                Deep speech representations extracted using
                facebook/wav2vec2-base.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Classification Pipeline")

    st.code(
        "Audio Input\n"
        "    ↓\n"
        "VAD Preprocessing\n"
        "    ↓\n"
        "343 Acoustic Features + 768 Wav2Vec2 Embeddings\n"
        "    ↓\n"
        "Combined 1,111-Dimensional Feature Vector\n"
        "    ↓\n"
        "StandardScaler\n"
        "    ↓\n"
        "SVM (RBF Kernel, C=10, class_weight=balanced)\n"
        "    ↓\n"
        "Emotion Prediction",
        language="text"
    )

    st.markdown("### Research Performance")

    performance = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Balanced Accuracy",
            "Macro F1-Score"
        ],
        "Result": [
            "61.12%",
            "61.21%",
            "61.18%"
        ]
    })

    st.dataframe(
        performance,
        use_container_width=True,
        hide_index=True
    )

    st.warning(
        "These are previously recorded experimental test results. "
        "They are not a confidence score for an individual prediction."
    )


# =========================================================
# PROJECT OVERVIEW PAGE
# =========================================================

elif page == "📊  Project Overview":

    st.markdown(
        '<div class="section-heading">Project Information</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="panel">
        <div class="eyebrow">PROJECT TITLE</div>
        <h2 style="color:#FFFFFF;font-family:Space Grotesk;">
            ML-Based Voice Personality Profiling and Conversational Mirror
        </h2>
        <p style="color:#AAB4D2;line-height:1.8;">
            This prototype demonstrates the emotion-recognition
            component of the project through speech signal processing,
            deep audio embeddings and machine learning classification.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Technologies Used")

    technologies = [
        ("Python", "Core programming language"),
        ("Librosa", "Audio processing and acoustic features"),
        ("WebRTC VAD", "Voice activity detection"),
        ("Wav2Vec2", "Deep speech representations"),
        ("Scikit-learn", "SVM classification and scaling"),
        ("Streamlit", "Interactive application interface"),
        ("Pandas & NumPy", "Feature data handling")
    ]

    for technology, purpose in technologies:
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom:10px;min-height:75px;">
            <b style="color:#C4B5FF;">{technology}</b>
            <span style="color:#AAB4D2;"> — {purpose}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Important Research Limitation")

    st.warning(
        "The current implemented model predicts six emotion classes. "
        "It does not yet perform Big Five personality regression. "
        "CREMA-D contains acted emotional speech, so real-world "
        "predictions should be interpreted cautiously."
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">
    <b style="color:#B9A5FF;">VOICEAI</b> ·
    ML-Based Voice Personality Profiling & Conversational Mirror
    <br><br>
    Research Prototype | Combined SVM | CREMA-D
</div>
""", unsafe_allow_html=True)