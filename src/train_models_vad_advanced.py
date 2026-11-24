
import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report
)

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


BASE_DIR = r"E:\VoiceEmotionRecognition_New"
FEATURE_DIR = os.path.join(BASE_DIR, "features")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# Load advanced VAD features
train_df = pd.read_csv(
    os.path.join(FEATURE_DIR, "train_features_vad_advanced.csv")
)

test_df = pd.read_csv(
    os.path.join(FEATURE_DIR, "test_features_vad_advanced.csv")
)

metadata_columns = ["filename", "emotion", "actor_id"]

feature_columns = [
    col for col in train_df.columns
    if col not in metadata_columns
]

# Ensure train and test have identical feature columns
assert feature_columns == [
    col for col in test_df.columns
    if col not in metadata_columns
], "Training and testing feature columns do not match!"

X_train = train_df[feature_columns].values
X_test = test_df[feature_columns].values

# Check numerical values
assert np.isfinite(X_train).all(), "Invalid values in training features"
assert np.isfinite(X_test).all(), "Invalid values in testing features"

label_encoder = LabelEncoder()

y_train = label_encoder.fit_transform(train_df["emotion"])
y_test = label_encoder.transform(test_df["emotion"])

print("\nADVANCED VAD DATASET")
print("--------------------")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Number of features:", len(feature_columns))
print("Emotion classes:", list(label_encoder.classes_))


models = {
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            class_weight="balanced"
        ))
    ]),

    "LightGBM": LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=31,
        class_weight="balanced",
        random_state=42,
        verbosity=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=4
    )
}


results = []

for name, model in models.items():

    print(f"\nTraining {name} with advanced VAD features...")

    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_accuracy = accuracy_score(
        y_train, train_predictions
    )

    test_accuracy = accuracy_score(
        y_test, test_predictions
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test, test_predictions
    )

    macro_f1 = f1_score(
        y_test, test_predictions,
        average="macro"
    )

    print(f"\n{name} RESULTS")
    print("--------------------")
    print(f"Training accuracy: {train_accuracy * 100:.2f}%")
    print(f"Testing accuracy: {test_accuracy * 100:.2f}%")
    print(f"Balanced accuracy: {balanced_accuracy * 100:.2f}%")
    print(f"Macro F1-score: {macro_f1 * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        test_predictions,
        labels=np.arange(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        zero_division=0
    ))

    model_filename = (
        f"{name.lower()}_ravdess_vad_advanced.pkl"
    )

    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "label_encoder": label_encoder
        },
        os.path.join(MODEL_DIR, model_filename)
    )

    results.append({
        "Model": name,
        "Train Accuracy": round(train_accuracy * 100, 2),
        "Test Accuracy": round(test_accuracy * 100, 2),
        "Balanced Accuracy": round(balanced_accuracy * 100, 2),
        "Macro F1": round(macro_f1 * 100, 2)
    })


# Save comparison results
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Test Accuracy",
    ascending=False
)

results_df.to_csv(
    os.path.join(
        REPORT_DIR,
        "vad_advanced_model_comparison.csv"
    ),
    index=False
)

with open(
    os.path.join(
        REPORT_DIR,
        "vad_advanced_model_comparison.json"
    ),
    "w"
) as file:
    json.dump(results, file, indent=4)


print("\n====================================")
print("ADVANCED VAD MODEL COMPARISON")
print("====================================")
print(results_df.to_string(index=False))

print("\nModels saved in:", MODEL_DIR)
print("Reports saved in:", REPORT_DIR)
print("\nADVANCED VAD TRAINING COMPLETED!")
