
import os
import json
import joblib
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report
)

print("\nAUGMENTED LIGHTGBM TRAINING")
print("--------------------------------")

# Load augmented training and original testing data
train_df = pd.read_csv(
    "features/train_features_vad_augmented.csv"
)

test_df = pd.read_csv(
    "features/test_features_vad.csv"
)

metadata = ["filename", "emotion", "actor_id"]

X_train = train_df.drop(columns=metadata)
y_train = train_df["emotion"]

X_test = test_df.drop(columns=metadata)
y_test = test_df["emotion"]

# Check feature consistency
if list(X_train.columns) != list(X_test.columns):
    raise ValueError("Training and testing features do not match!")

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Features:", X_train.shape[1])

# LightGBM model
model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=31,
    class_weight="balanced",
    random_state=42,
    n_jobs=4,
    verbosity=-1
)

print("\nTraining LightGBM...")
model.fit(X_train, y_train)

# Predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Evaluation
results = {
    "model": "LightGBM_Augmented",
    "training_samples": len(X_train),
    "testing_samples": len(X_test),
    "features": X_train.shape[1],

    "training_accuracy": float(
        accuracy_score(y_train, train_predictions)
    ),

    "testing_accuracy": float(
        accuracy_score(y_test, test_predictions)
    ),

    "balanced_accuracy": float(
        balanced_accuracy_score(y_test, test_predictions)
    ),

    "macro_f1": float(
        f1_score(y_test, test_predictions, average="macro")
    ),

    "classification_report": classification_report(
        y_test,
        test_predictions,
        output_dict=True
    )
}

print("\nRESULTS")
print("--------------------------------")

print(
    "Training Accuracy:",
    f"{results['training_accuracy'] * 100:.2f}%"
)

print(
    "Testing Accuracy:",
    f"{results['testing_accuracy'] * 100:.2f}%"
)

print(
    "Balanced Accuracy:",
    f"{results['balanced_accuracy'] * 100:.2f}%"
)

print(
    "Macro F1:",
    f"{results['macro_f1'] * 100:.2f}%"
)

print("\nCLASSIFICATION REPORT")
print(
    classification_report(
        y_test,
        test_predictions
    )
)

# Save model and results separately
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

joblib.dump(
    model,
    "models/lightgbm_ravdess_augmented.pkl"
)

with open(
    "reports/augmented_lightgbm_results.json",
    "w"
) as file:
    json.dump(results, file, indent=4)

print("\nModel saved successfully.")
print("Results saved successfully.")
