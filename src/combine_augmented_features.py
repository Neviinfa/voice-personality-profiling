
import pandas as pd
import os

print("\nCOMBINING ORIGINAL + AUGMENTED TRAINING FEATURES")
print("------------------------------------------------")

original = pd.read_csv(
    "features/train_features_vad.csv"
)

augmented = pd.read_csv(
    "features/augmented_train_features_vad.csv"
)

test = pd.read_csv(
    "features/test_features_vad.csv"
)

combined = pd.concat(
    [original, augmented],
    ignore_index=True
)

# Verify feature consistency
if list(original.columns) != list(augmented.columns):
    raise ValueError("Feature columns do not match!")

if list(original.columns) != list(test.columns):
    raise ValueError("Test feature columns do not match!")

if combined.isnull().any().any():
    raise ValueError("Missing values detected!")

combined.to_csv(
    "features/train_features_vad_augmented.csv",
    index=False
)

print("Original training samples:", len(original))
print("Augmented samples:", len(augmented))
print("Combined training samples:", len(combined))
print("Testing samples:", len(test))
print("Feature count:", len(combined.columns) - 3)

print("\nEmotion distribution:")
print(combined["emotion"].value_counts())

print("\nSaved: features/train_features_vad_augmented.csv")
print("Original test dataset remains unchanged.")
