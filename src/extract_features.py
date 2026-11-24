
from pathlib import Path
import warnings

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = ROOT / "features" / "train.csv"
TEST_CSV = ROOT / "features" / "test.csv"

OUTPUT_TRAIN = ROOT / "features" / "train_features.csv"
OUTPUT_TEST = ROOT / "features" / "test_features.csv"

def extract_features(filepath):
    y, sr = librosa.load(filepath, sr=22050, mono=True)

    features = {}

    # MFCC: 13 coefficients and their statistics
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f"mfcc_{i+1}_mean"] = np.mean(mfcc[i])
        features[f"mfcc_{i+1}_std"] = np.std(mfcc[i])

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features["chroma_mean"] = np.mean(chroma)
    features["chroma_std"] = np.std(chroma)

    # Spectral features
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    for name, value in [
        ("centroid", centroid),
        ("bandwidth", bandwidth),
        ("rolloff", rolloff),
        ("zcr", zcr),
        ("rms", rms),
    ]:
        features[f"{name}_mean"] = np.mean(value)
        features[f"{name}_std"] = np.std(value)

    # Pitch statistics
    pitches = librosa.yin(y, fmin=65, fmax=500, sr=sr)
    features["pitch_mean"] = np.mean(pitches)
    features["pitch_std"] = np.std(pitches)

    return features


def process_split(csv_path, output_path):
    df = pd.read_csv(csv_path)
    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            feature_values = extract_features(row["filepath"])
            feature_values["filename"] = row["filename"]
            feature_values["emotion"] = row["emotion"]
            feature_values["actor_id"] = row["actor_id"]
            rows.append(feature_values)

        except Exception as error:
            print(f"Failed: {row['filename']} - {error}")

    result = pd.DataFrame(rows)
    result.to_csv(output_path, index=False)

    print(f"\nSaved: {output_path}")
    print("Successful recordings:", len(result))
    print("Feature count:", len(result.columns) - 3)


if __name__ == "__main__":
    process_split(TRAIN_CSV, OUTPUT_TRAIN)
    process_split(TEST_CSV, OUTPUT_TEST)
    print("\nFEATURE EXTRACTION COMPLETED")
