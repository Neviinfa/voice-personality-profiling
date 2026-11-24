
import os
import sys
import numpy as np
import pandas as pd
import soundfile as sf
import torch

from scipy.signal import resample_poly
from transformers import AutoProcessor, AutoModel

# Project configuration
BASE_DIR = r"E:\VoiceEmotionRecognition_New"
FEATURE_DIR = os.path.join(BASE_DIR, "features")

MODEL_NAME = "facebook/wav2vec2-base"
SAMPLE_RATE = 16000

TRAIN_CSV = os.path.join(FEATURE_DIR, "train.csv")
TEST_CSV = os.path.join(FEATURE_DIR, "test.csv")

TRAIN_OUTPUT = os.path.join(
    FEATURE_DIR, "train_wav2vec2_embeddings.csv"
)
TEST_OUTPUT = os.path.join(
    FEATURE_DIR, "test_wav2vec2_embeddings.csv"
)

# Prevent accidental overwriting
if os.path.exists(TRAIN_OUTPUT) or os.path.exists(TEST_OUTPUT):
    print("Output files already exist. Stopping to protect existing results.")
    sys.exit(1)

torch.set_num_threads(2)
device = torch.device("cpu")

print("Loading pretrained Wav2Vec2 model...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

def extract_embedding(filepath):
    audio, sr = sf.read(filepath)

    # Convert stereo audio to mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    audio = audio.astype(np.float32)

    # Resample to 16 kHz
    if sr != SAMPLE_RATE:
        audio = resample_poly(
            audio,
            SAMPLE_RATE,
            sr
        ).astype(np.float32)

    inputs = processor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model(**inputs)

    # Mean pooling across time frames
    embedding = outputs.last_hidden_state.mean(dim=1)

    return embedding.squeeze(0).cpu().numpy()


def process_split(csv_path, output_path, split_name):
    df = pd.read_csv(csv_path)

    required = ["filename", "filepath", "emotion", "actor_id"]
    for column in required:
        if column not in df.columns:
            raise ValueError(f"Missing column: {column}")

    rows = []

    print(f"\nProcessing {split_name}: {len(df)} recordings")

    for index, row in df.iterrows():
        filepath = row["filepath"]

        if not os.path.isfile(filepath):
            raise FileNotFoundError(filepath)

        embedding = extract_embedding(filepath)

        result = {
            "filename": row["filename"],
            "emotion": row["emotion"],
            "actor_id": row["actor_id"]
        }

        for i, value in enumerate(embedding):
            result[f"embedding_{i:03d}"] = float(value)

        rows.append(result)

        if (index + 1) % 25 == 0 or index + 1 == len(df):
            print(
                f"{split_name}: {index + 1}/{len(df)} completed"
            )

    output_df = pd.DataFrame(rows)
    output_df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print("Output shape:", output_df.shape)


if __name__ == "__main__":
    process_split(TRAIN_CSV, TRAIN_OUTPUT, "TRAIN")
    process_split(TEST_CSV, TEST_OUTPUT, "TEST")

    print("\nWav2Vec2 extraction completed successfully.")
