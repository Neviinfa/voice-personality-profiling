
import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report
)

BASE_DIR = r"E:\VoiceEmotionRecognition_New"
FEATURE_DIR = os.path.join(BASE_DIR, "features")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


def load_fused_data(split):
    vad_path = os.path.join(
        FEATURE_DIR, f"{split}_features_vad.csv"
    )
    prosodic_path = os.path.join(
        FEATURE_DIR, f"{split}_features_vad_prosodic.csv"
    )

    vad = pd.read_csv(vad_path)
    prosodic = pd.read_csv(prosodic_path)

    # Confirm matching recordings and labels
    if vad["filename"].duplicated().any():
        raise ValueError(f"Duplicate VAD filenames in {split}")

    if prosodic["filename"].duplicated().any():
        raise ValueError(f"Duplicate prosodic filenames in {split}")

    if set(vad["filename"]) != set(prosodic["filename"]):
        raise ValueError(f"Filename mismatch in {split}")

    vad = vad.sort_values("filename").reset_index(drop=True)
    prosodic = prosodic.sort_values("filename").reset_index(drop=True)

    if not np.array_equal(vad["emotion"], prosodic["emotion"]):
        raise ValueError(f"Emotion labels mismatch in {split}")

    if not np.array_equal(vad["actor_id"], prosodic["actor_id"]):
        raise ValueError(f"Actor IDs mismatch in {split}")

    metadata = vad[["filename", "emotion", "actor_id"]].copy()

    vad_features = vad.select_dtypes(include=np.number).copy()
    prosodic_features = prosodic.select_dtypes(include=np.number).copy()

    vad_features.columns = [
        f"vad_{column}" for column in vad_features.columns
    ]
    prosodic_features.columns = [
        f"prosodic_{column}" for column in prosodic_features.columns
    ]

    X = pd.concat(
        [vad_features, prosodic_features], axis=1
    )

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError(f"Non-finite features found in {split}")

    return X, metadata["emotion"], metadata


print("\nVAD + PROSODIC FEATURE FUSION")
print("--------------------------------")

X_train, y_train, train_meta = load_fused_data("train")
X_test, y_test, test_meta = load_fused_data("test")

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Combined features:", X_train.shape[1])

if list(X_train.columns) != list(X_test.columns):
    raise ValueError("Training and testing feature columns differ")

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA()),
    ("svm", SVC(
        kernel="rbf",
        class_weight="balanced"
    ))
])

param_grid = {
    "pca__n_components": [20, 30, 40, 50, 60, 70, 80, 90],
    "svm__C": [1, 10, 100],
    "svm__gamma": ["scale", 0.01, 0.001]
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

search = GridSearchCV(
    pipeline,
    param_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

print("\nStarting PCA + SVM optimization...")
search.fit(X_train, y_train)

best_model = search.best_estimator_

train_predictions = best_model.predict(X_train)
test_predictions = best_model.predict(X_test)

train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)
balanced = balanced_accuracy_score(y_test, test_predictions)
macro_f1 = f1_score(y_test, test_predictions, average="macro")

print("\nBEST PARAMETERS")
print(search.best_params_)
print("\nBEST CV MACRO-F1:", round(search.best_score_, 4))

print("\nFINAL RESULTS")
print("--------------------------------")
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")
print(f"Balanced Accuracy: {balanced * 100:.2f}%")
print(f"Macro F1: {macro_f1 * 100:.2f}%")

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, test_predictions))

model_path = os.path.join(
    MODEL_DIR, "svm_ravdess_vad_prosodic_fusion.pkl"
)

joblib.dump({
    "model": best_model,
    "feature_columns": list(X_train.columns),
    "classes": list(best_model.classes_),
    "best_params": search.best_params_
}, model_path)

results = {
    "training_accuracy": float(train_accuracy),
    "testing_accuracy": float(test_accuracy),
    "balanced_accuracy": float(balanced),
    "macro_f1": float(macro_f1),
    "best_cv_macro_f1": float(search.best_score_),
    "best_params": search.best_params_,
    "feature_count": int(X_train.shape[1]),
    "training_samples": int(len(X_train)),
    "testing_samples": int(len(X_test))
}

report_path = os.path.join(
    REPORT_DIR, "fusion_pca_svm_results.json"
)

with open(report_path, "w") as file:
    json.dump(results, file, indent=4)

print("\nModel saved:", model_path)
print("Results saved:", report_path)
print("\nFEATURE FUSION EXPERIMENT COMPLETED!")
