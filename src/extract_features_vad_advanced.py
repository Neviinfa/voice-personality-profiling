
import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import webrtcvad

from tqdm import tqdm

BASE_DIR = r"E:\VoiceEmotionRecognition_New"
FEATURE_DIR = os.path.join(BASE_DIR, "features")

SR = 16000
FRAME_DURATION = 30
VAD_MODE = 2

vad = webrtcvad.Vad(VAD_MODE)


def detect_speech(audio, original_sr):
    audio = librosa.resample(
        audio.astype(np.float32),
        orig_sr=original_sr,
        target_sr=SR
    )

    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)

    frame_size = int(SR * FRAME_DURATION / 1000)
    usable_length = (len(pcm) // frame_size) * frame_size
    pcm = pcm[:usable_length]

    speech_frames = []

    for start in range(0, len(pcm), frame_size):
        frame = pcm[start:start + frame_size]

        if vad.is_speech(frame.tobytes(), SR):
            speech_frames.append(frame)

    if not speech_frames:
        return np.array([], dtype=np.float32)

    return np.concatenate(speech_frames).astype(np.float32) / 32768.0


def extract_advanced_features(audio):
    features = {}

    # 1. MFCC, delta and delta-delta
    mfcc = librosa.feature.mfcc(
        y=audio, sr=SR, n_mfcc=13,
        n_fft=1024, hop_length=256
    )

    delta = librosa.feature.delta(mfcc, width=5, order=1)
    delta2 = librosa.feature.delta(mfcc, width=5, order=2)

    for name, matrix in [
        ("mfcc", mfcc),
        ("delta", delta),
        ("delta2", delta2)
    ]:
        for i in range(13):
            features[f"{name}_{i+1}_mean"] = float(np.mean(matrix[i]))
            features[f"{name}_{i+1}_std"] = float(np.std(matrix[i]))

    # 2. Spectral contrast
    contrast = librosa.feature.spectral_contrast(
        y=audio, sr=SR, n_fft=1024,
        hop_length=256
    )

    for i in range(contrast.shape[0]):
        features[f"contrast_{i+1}_mean"] = float(np.mean(contrast[i]))
        features[f"contrast_{i+1}_std"] = float(np.std(contrast[i]))

    # 3. Spectral flatness
    flatness = librosa.feature.spectral_flatness(
        y=audio, n_fft=1024, hop_length=256
    )[0]

    features["flatness_mean"] = float(np.mean(flatness))
    features["flatness_std"] = float(np.std(flatness))

    # 4. Pitch statistics
    pitch = librosa.yin(
        audio, fmin=65, fmax=500,
        sr=SR, frame_length=1024, hop_length=256
    )

    features["pitch_mean"] = float(np.mean(pitch))
    features["pitch_std"] = float(np.std(pitch))
    features["pitch_median"] = float(np.median(pitch))
    features["pitch_10th"] = float(np.percentile(pitch, 10))
    features["pitch_90th"] = float(np.percentile(pitch, 90))

    # 5. Energy statistics
    rms = librosa.feature.rms(
        y=audio, frame_length=1024, hop_length=256
    )[0]

    features["energy_mean"] = float(np.mean(rms))
    features["energy_std"] = float(np.std(rms))
    features["energy_max"] = float(np.max(rms))

    return features


def process_split(split_name):
    input_csv = os.path.join(FEATURE_DIR, f"{split_name}.csv")
    output_csv = os.path.join(
        FEATURE_DIR,
        f"{split_name}_features_vad_advanced.csv"
    )

    metadata = pd.read_csv(input_csv)
    results = []
    failed = 0
    skipped = 0

    print(f"\nProcessing {split_name}: {len(metadata)} recordings")

    for _, row in tqdm(metadata.iterrows(), total=len(metadata)):
        try:
            audio, original_sr = sf.read(row["filepath"])

            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)

            speech_audio = detect_speech(audio, original_sr)

            if len(speech_audio) < int(0.25 * SR):
                skipped += 1
                continue

            features = extract_advanced_features(speech_audio)

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
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    process_split("train")
    process_split("test")

    print("\nADVANCED VAD FEATURE EXTRACTION COMPLETED!")
