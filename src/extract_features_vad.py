
import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import webrtcvad

from tqdm import tqdm

BASE_DIR = r"E:\VoiceEmotionRecognition_New"
FEATURE_DIR = os.path.join(BASE_DIR, "features")

SAMPLE_RATE = 16000
FRAME_DURATION = 30
VAD_MODE = 2

vad = webrtcvad.Vad(VAD_MODE)


def detect_speech(audio, sr=SAMPLE_RATE):
    """Detect speech frames using WebRTC VAD."""

    audio = librosa.resample(
        audio, orig_sr=sr, target_sr=SAMPLE_RATE
    )

    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)

    frame_size = int(SAMPLE_RATE * FRAME_DURATION / 1000)
    frame_bytes = frame_size * 2

    usable_length = (len(pcm) // frame_size) * frame_size
    pcm = pcm[:usable_length]

    speech_frames = []

    for start in range(0, len(pcm), frame_size):
        frame = pcm[start:start + frame_size]
        frame_data = frame.tobytes()

        if vad.is_speech(frame_data, SAMPLE_RATE):
            speech_frames.append(frame)

    if not speech_frames:
        return np.array([], dtype=np.float32)

    speech_audio = np.concatenate(speech_frames)
    return speech_audio.astype(np.float32) / 32768.0


def extract_features(audio, sr=SAMPLE_RATE):
    """Extract 40 acoustic features from VAD-filtered speech."""

    features = {}

    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=13
    )

    for i in range(13):
        features[f"mfcc_{i+1}_mean"] = np.mean(mfcc[i])
        features[f"mfcc_{i+1}_std"] = np.std(mfcc[i])

    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)

    features["chroma_mean"] = np.mean(chroma)
    features["chroma_std"] = np.std(chroma)

    spectral_features = {
        "spectral_centroid": librosa.feature.spectral_centroid(
            y=audio, sr=sr
        )[0],
        "spectral_bandwidth": librosa.feature.spectral_bandwidth(
            y=audio, sr=sr
        )[0],
        "spectral_rolloff": librosa.feature.spectral_rolloff(
            y=audio, sr=sr
        )[0],
        "zero_crossing_rate": librosa.feature.zero_crossing_rate(
            y=audio
        )[0],
        "rms": librosa.feature.rms(y=audio)[0],
    }

    for name, values in spectral_features.items():
        features[f"{name}_mean"] = np.mean(values)
        features[f"{name}_std"] = np.std(values)

    pitches = librosa.yin(
        audio, fmin=65, fmax=500, sr=sr
    )

    features["pitch_mean"] = np.mean(pitches)
    features["pitch_std"] = np.std(pitches)

    return features


def process_split(split_name):
    input_csv = os.path.join(
        FEATURE_DIR, f"{split_name}.csv"
    )

    output_csv = os.path.join(
        FEATURE_DIR, f"{split_name}_features_vad.csv"
    )

    metadata = pd.read_csv(input_csv)
    results = []
    failed = 0
    skipped = 0

    print(f"\nProcessing {split_name}: {len(metadata)} recordings")

    for _, row in tqdm(metadata.iterrows(), total=len(metadata)):
        try:
            audio, sr = sf.read(row["filepath"])

            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)

            audio = librosa.resample(
                audio.astype(np.float32),
                orig_sr=sr,
                target_sr=SAMPLE_RATE
            )

            speech_audio = detect_speech(audio)

            if len(speech_audio) < int(0.25 * SAMPLE_RATE):
                skipped += 1
                continue

            features = extract_features(speech_audio)

            features["filename"] = row["filename"]
            features["emotion"] = row["emotion"]
            features["actor_id"] = row["actor_id"]

            results.append(features)

        except Exception as error:
            failed += 1
            print(f"\nError: {row['filename']} - {error}")

    output = pd.DataFrame(results)

    output.to_csv(output_csv, index=False)

    print(f"\nSaved: {output_csv}")
    print(f"Successful: {len(results)}")
    print(f"Skipped (insufficient speech): {skipped}")
    print(f"Failed: {failed}")
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    process_split("train")
    process_split("test")

    print("\nVAD FEATURE EXTRACTION COMPLETED!")
