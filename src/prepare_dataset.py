
from pathlib import Path
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
DATASET = PROJECT / "dataset" / "Audio_Speech_Actors_01-24"
OUTPUT = PROJECT / "features" / "ravdess_metadata.csv"

emotion_labels = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

records = []

for audio_file in sorted(DATASET.rglob("*.wav")):
    parts = audio_file.stem.split("-")

    if len(parts) != 7:
        print("Skipping invalid filename:", audio_file.name)
        continue

    modality, vocal_channel, emotion, intensity, statement, repetition, actor = parts

    if emotion not in emotion_labels:
        print("Skipping unknown emotion:", audio_file.name)
        continue

    records.append({
        "filename": audio_file.name,
        "filepath": str(audio_file.resolve()),
        "emotion": emotion_labels[emotion],
        "emotion_code": emotion,
        "actor_id": int(actor),
        "intensity": intensity,
        "statement": statement,
        "repetition": repetition,
    })

df = pd.DataFrame(records)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

print("RAVDESS METADATA CREATED")
print("Total recordings:", len(df))
print("\nEmotion distribution:")
print(df["emotion"].value_counts().sort_index())
print("\nUnique actors:", df["actor_id"].nunique())
print("Saved to:", OUTPUT)

if len(df) != 1440:
    raise ValueError(f"Expected 1440 recordings, found {len(df)}")

if df["actor_id"].nunique() != 24:
    raise ValueError("Expected 24 actors")
