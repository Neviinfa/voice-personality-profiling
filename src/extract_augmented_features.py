
import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import webrtcvad

from tqdm import tqdm

BASE_DIR = r"E:\VoiceEmotionRecognition_New"

AUGMENTED_DIR = os.path.join(
    BASE_DIR, "dataset", "augmented_train"
)

ORIGINAL_TRAIN_CSV = os.path.join(
    BASE_DIR, "features", "train_features_vad.csv"
)

OUTPUT_CSV = os.path.join(
    BASE_DIR, "features", "augmented_train_features_vad.csv"
)

SAMPLE_RATE = 16000
FRAME_DURATION = 30
VAD_MODE = 2

vad = webrtcvad.Vad(VAD_MODE)


def detect_speech(audio, sr=SAMPLE_RATE):

    audio = np.clip(audio, -1.0, 1.0)

    pcm = (audio * 32767).astype(np.int16)

    frame_size = int(
        SAMPLE_RATE * FRAME_DURATION / 1000
    )

    usable_length = (
        len(pcm) // frame_size
    ) * frame_size

    pcm = pcm[:usable_length]

    speech_frames = []

    for start in range(0, len(pcm), frame_size):

        frame = pcm[start:start + frame_size]

        if vad.is_speech(
            frame.tobytes(), SAMPLE_RATE
        ):
            speech_frames.append(frame)

    if not speech_frames:
        return np.array([], dtype=np.float32)

    speech_audio = np.concatenate(speech_frames)

    return speech_audio.astype(np.float32) / 32768.0


def extract_features(audio, sr=SAMPLE_RATE):

    features = {}

    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=13
    )

    for i in range(13):
        features[f"mfcc_{i+1}_mean"] = np.mean(mfcc[i])
        features[f"mfcc_{i+1}_std"] = np.std(mfcc[i])

    chroma = librosa.feature.chroma_stft(
        y=audio, sr=sr
    )

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

        "rms": librosa.feature.rms(
            y=audio
        )[0]
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


def main():

    original_df = pd.read_csv(ORIGINAL_TRAIN_CSV)

    original_labels = original_df.set_index(
        "filename"
    )[["emotion", "actor_id"]]

    files = sorted([
        f for f in os.listdir(AUGMENTED_DIR)
        if f.lower().endswith(".wav")
    ])

    results = []
    failed = 0
    skipped = 0

    print("\nAUGMENTED VAD FEATURE EXTRACTION")
    print("--------------------------------")
    print("Augmented recordings:", len(files))

    for filename in tqdm(files):

        try:

            # Original RAVDESS filename is the part
            # before the augmentation suffix.
            base_name = filename.rsplit("_", 1)[0] + ".wav"

            if base_name not in original_labels.index:
                failed += 1
                print("\nMetadata not found:", filename)
                continue

            emotion = original_labels.loc[
                base_name, "emotion"
            ]

            actor_id = original_labels.loc[
                base_name, "actor_id"
            ]

            filepath = os.path.join(
                AUGMENTED_DIR, filename
            )

            audio, sr = sf.read(filepath)

            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)

            audio = audio.astype(np.float32)

            if sr != SAMPLE_RATE:
                audio = librosa.resample(
                    audio,
                    orig_sr=sr,
                    target_sr=SAMPLE_RATE
                )

            speech_audio = detect_speech(audio)

            if len(speech_audio) < int(
                0.25 * SAMPLE_RATE
            ):
                skipped += 1
                continue

            features = extract_features(speech_audio)

            features["filename"] = filename
            features["emotion"] = emotion
            features["actor_id"] = actor_id

            results.append(features)

        except Exception as error:

            failed += 1
            print(
                f"\nError: {filename} - {error}"
            )

    output = pd.DataFrame(results)

    # Keep the same feature and metadata column order
    original_columns = original_df.columns.tolist()

    feature_columns = [
        col for col in original_columns
        if col not in ["filename", "emotion", "actor_id"]
    ]

    expected_columns = (
        feature_columns +
        ["filename", "emotion", "actor_id"]
    )

    output = output.reindex(
        columns=expected_columns
    )

    output.to_csv(OUTPUT_CSV, index=False)

    print("\nFEATURE EXTRACTION COMPLETED")
    print("--------------------------------")
    print("Total files:", len(files))
    print("Successful:", len(results))
    print("Skipped:", skipped)
    print("Failed:", failed)
    print("Output shape:", output.shape)
    print("Output saved:", OUTPUT_CSV)

    print(
        "\nFeature count:",
        len(feature_columns)
    )


if __name__ == "__main__":
    main()
