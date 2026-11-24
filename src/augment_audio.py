
import os
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from tqdm import tqdm

SOURCE_DIR = "dataset/Audio_Speech_Actors_01-24"
OUTPUT_DIR = "dataset/augmented_train"
TRAIN_CSV = "features/train_features_vad.csv"

SAMPLE_RATE = 16000
RANDOM_SEED = 42

rng = np.random.default_rng(RANDOM_SEED)

train_df = pd.read_csv(TRAIN_CSV)

os.makedirs(OUTPUT_DIR, exist_ok=True)

def augment_audio(audio, method):
    if method == "noise":
        noise_level = rng.uniform(0.002, 0.008)
        noise = rng.normal(0, 1, len(audio))
        augmented = audio + noise_level * noise

    elif method == "shift":
        shift = int(rng.uniform(-0.10, 0.10) * len(audio))
        augmented = np.roll(audio, shift)

    elif method == "speed":
        rate = rng.uniform(0.95, 1.05)
        augmented = librosa.effects.time_stretch(
            audio.astype(np.float32),
            rate=rate
        )

    else:
        raise ValueError("Unknown augmentation method")

    return np.clip(augmented, -1.0, 1.0)

methods = ["noise", "shift", "speed"]

created = 0
failed = 0

print("Starting training-only audio augmentation...")
print("Original training recordings:", len(train_df))

for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
    filename = row["filename"]

    # Find original recording
    source_path = None

    for root, _, files in os.walk(SOURCE_DIR):
        if filename in files:
            source_path = os.path.join(root, filename)
            break

    if source_path is None:
        print("Missing:", filename)
        failed += 1
        continue

    try:
        audio, sr = sf.read(source_path)

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        if sr != SAMPLE_RATE:
            audio = librosa.resample(
                audio.astype(np.float32),
                orig_sr=sr,
                target_sr=SAMPLE_RATE
            )

        audio = audio.astype(np.float32)

        for method in methods:
            augmented = augment_audio(audio, method)

            output_name = (
                os.path.splitext(filename)[0]
                + "_" + method + ".wav"
            )

            output_path = os.path.join(
                OUTPUT_DIR,
                output_name
            )

            sf.write(output_path, augmented, SAMPLE_RATE)
            created += 1

    except Exception as e:
        print(f"\nFailed {filename}: {e}")
        failed += 1

print("\nAUGMENTATION COMPLETED")
print("Original training recordings:", len(train_df))
print("Augmented files created:", created)
print("Failed recordings:", failed)
print("Output folder:", OUTPUT_DIR)
