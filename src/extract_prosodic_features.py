
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


def detect_speech(audio):
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)

    frame_size = int(SAMPLE_RATE * FRAME_DURATION / 1000)
    usable_length = (len(pcm) // frame_size) * frame_size
    pcm = pcm[:usable_length]

    speech_frames = []

    for start in range(0, len(pcm), frame_size):
        frame = pcm[start:start + frame_size]

        if vad.is_speech(frame.tobytes(), SAMPLE_RATE):
            speech_frames.append(frame)

    if not speech_frames:
        return np.array([], dtype=np.float32)

    return np.concatenate(speech_frames).astype(
        np.float32
    ) / 32768.0


def extract_features(audio):

    features = {}

    # 1. Existing MFCC features
    mfcc = librosa.feature.mfcc(
        y=audio, sr=SAMPLE_RATE, n_mfcc=13
    )

    for i in range(13):
        features[f"mfcc_{i+1}_mean"] = np.mean(mfcc[i])
        features[f"mfcc_{i+1}_std"] = np.std(mfcc[i])

    # 2. Existing chroma features
    chroma = librosa.feature.chroma_stft(
        y=audio, sr=SAMPLE_RATE
    )

    features["chroma_mean"] = np.mean(chroma)
    features["chroma_std"] = np.std(chroma)

    # 3. Existing spectral and energy features
    spectral_features = {
        "spectral_centroid": librosa.feature.spectral_centroid(
            y=audio, sr=SAMPLE_RATE
        )[0],
        "spectral_bandwidth": librosa.feature.spectral_bandwidth(
            y=audio, sr=SAMPLE_RATE
        )[0],
        "spectral_rolloff": librosa.feature.spectral_rolloff(
            y=audio, sr=SAMPLE_RATE
        )[0],
        "zero_crossing_rate": librosa.feature.zero_crossing_rate(
            y=audio
        )[0],
        "rms": librosa.feature.rms(y=audio)[0]
    }

    for name, values in spectral_features.items():
        features[f"{name}_mean"] = np.mean(values)
        features[f"{name}_std"] = np.std(values)

    # 4. Existing pitch features
    pitches = librosa.yin(
        audio, fmin=65, fmax=500, sr=SAMPLE_RATE
    )

    features["pitch_mean"] = np.mean(pitches)
    features["pitch_std"] = np.std(pitches)

    # 5. New pitch statistics
    features["pitch_median"] = np.median(pitches)
    features["pitch_min"] = np.min(pitches)
    features["pitch_max"] = np.max(pitches)
    features["pitch_range"] = np.ptp(pitches)

    features["pitch_q25"] = np.percentile(pitches, 25)
    features["pitch_q75"] = np.percentile(pitches, 75)

    # 6. New energy statistics
    rms = spectral_features["rms"]

    features["energy_median"] = np.median(rms)
    features["energy_q25"] = np.percentile(rms, 25)
    features["energy_q75"] = np.percentile(rms, 75)
    features["energy_range"] = np.ptp(rms)

    return features


def process_split(split_name):

    input_csv = os.path.join(
        FEATURE_DIR, f"{split_name}.csv"
    )

    output_csv = os.path.join(
        FEATURE_DIR,
        f"{split_name}_features_vad_prosodic.csv"
    )

    metadata = pd.read_csv(input_csv)

    results = []
    failed = 0
    skipped = 0

    print(
        f"\nProcessing {split_name}: {len(metadata)} recordings"
    )

    for _, row in tqdm(
        metadata.iterrows(), total=len(metadata)
    ):

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
            print(
                f"\nError: {row['filename']} - {error}"
            )

    output = pd.DataFrame(results)

    metadata_columns = [
        "filename", "emotion", "actor_id"
    ]

    feature_columns = [
        col for col in output.columns
        if col not in metadata_columns
    ]

    output = output[
        feature_columns + metadata_columns
    ]

    output.to_csv(output_csv, index=False)

    print("\nSaved:", output_csv)
    print("Successful:", len(results))
    print("Skipped:", skipped)
    print("Failed:", failed)
    print("Output shape:", output.shape)
    print("Feature count:", len(feature_columns))


if __name__ == "__main__":

    process_split("train")
    process_split("test")

    print("\nPROSODIC FEATURE EXTRACTION COMPLETED!")
