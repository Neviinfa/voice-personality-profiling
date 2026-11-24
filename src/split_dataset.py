
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]

metadata_path = ROOT / "features" / "ravdess_metadata.csv"
train_path = ROOT / "features" / "train.csv"
test_path = ROOT / "features" / "test.csv"

df = pd.read_csv(metadata_path)

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["emotion"]
)

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print("DATASET SPLIT COMPLETED")
print("Training samples:", len(train_df))
print("Testing samples:", len(test_df))

print("\nTraining distribution:")
print(train_df["emotion"].value_counts().sort_index())

print("\nTesting distribution:")
print(test_df["emotion"].value_counts().sort_index())

print("\nTraining actors:", train_df["actor_id"].nunique())
print("Testing actors:", test_df["actor_id"].nunique())
print("Speaker overlap:", len(
    set(train_df["actor_id"]) & set(test_df["actor_id"])
))
