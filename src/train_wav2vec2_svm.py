
import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# =====================================================
# CONFIGURATION
# =====================================================

BASE_DIR = r"E:\VoiceEmotionRecognition_New"
FEATURE_DIR = os.path.join(BASE_DIR, "features")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

TRAIN_FILE = os.path.join(
    FEATURE_DIR, "train_wav2vec2_embeddings.csv"
)

TEST_FILE = os.path.join(
    FEATURE_DIR, "test_wav2vec2_embeddings.csv"
)

MODEL_OUTPUT = os.path.join(
    MODEL_DIR, "svm_ravdess_wav2vec2_pca.pkl"
)

RESULT_OUTPUT = os.path.join(
    REPORT_DIR, "wav2vec2_svm_results.json"
)

CV_OUTPUT = os.path.join(
    REPORT_DIR, "wav2vec2_svm_cv_results.csv"
)

# =====================================================
# LOAD DATA
# =====================================================

print("=" * 60)
print("WAV2VEC2 + PCA + SVM EXPERIMENT")
print("=" * 60)

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

embedding_columns = [
    col for col in train_df.columns
    if col.startswith("embedding_")
]

test_embedding_columns = [
    col for col in test_df.columns
    if col.startswith("embedding_")
]

if len(embedding_columns) != 768:
    raise ValueError(
        f"Expected 768 training embeddings, got {len(embedding_columns)}"
    )

if embedding_columns != test_embedding_columns:
    raise ValueError("Training and testing embedding columns do not match.")

if train_df["emotion"].isna().any() or test_df["emotion"].isna().any():
    raise ValueError("Missing emotion labels detected.")

if not train_df["filename"].is_unique:
    raise ValueError("Duplicate training filenames detected.")

if not test_df["filename"].is_unique:
    raise ValueError("Duplicate testing filenames detected.")

if set(train_df["filename"]) & set(test_df["filename"]):
    raise ValueError("Train/test filename overlap detected.")

X_train = train_df[embedding_columns].to_numpy(dtype=np.float32)
X_test = test_df[embedding_columns].to_numpy(dtype=np.float32)

y_train = train_df["emotion"].to_numpy()
y_test = test_df["emotion"].to_numpy()

if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
    raise ValueError("NaN or infinite embedding values detected.")

if set(y_train) != set(y_test):
    raise ValueError("Training and testing emotion classes do not match.")

print("\nTraining shape:", X_train.shape)
print("Testing shape:", X_test.shape)
print("Classes:", sorted(set(y_train)))

# =====================================================
# PIPELINE
# =====================================================

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(random_state=42)),
    ("svm", SVC())
])

# =====================================================
# HYPERPARAMETER SEARCH
# =====================================================

param_grid = {
    "pca__n_components": [50, 75, 100, 150, 200],
    "svm__C": [0.1, 1, 10, 100],
    "svm__gamma": ["scale", 0.001, 0.01],
    "svm__kernel": ["rbf"]
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=1,
    verbose=2,
    refit=True,
    error_score="raise"
)

print("\nStarting 300 candidate combinations (1500 fits)...")

search.fit(X_train, y_train)

best_model = search.best_estimator_

print("\nBest parameters:")
print(search.best_params_)

print("\nBest CV Macro F1:", round(search.best_score_, 4))

# =====================================================
# EVALUATION
# =====================================================

train_predictions = best_model.predict(X_train)
test_predictions = best_model.predict(X_test)

train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)

balanced_acc = balanced_accuracy_score(
    y_test, test_predictions
)

macro_f1 = f1_score(
    y_test, test_predictions, average="macro"
)

weighted_f1 = f1_score(
    y_test, test_predictions, average="weighted"
)

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(f"Training Accuracy:   {train_accuracy * 100:.2f}%")
print(f"Testing Accuracy:    {test_accuracy * 100:.2f}%")
print(f"Balanced Accuracy:   {balanced_acc * 100:.2f}%")
print(f"Testing Macro F1:    {macro_f1 * 100:.2f}%")
print(f"Testing Weighted F1: {weighted_f1 * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        test_predictions,
        zero_division=0
    )
)

# =====================================================
# SAVE MODEL
# =====================================================

joblib.dump(
    {
        "model": best_model,
        "embedding_columns": embedding_columns,
        "model_name": "facebook/wav2vec2-base",
        "best_parameters": search.best_params_,
        "classes": sorted(set(y_train))
    },
    MODEL_OUTPUT
)

# =====================================================
# SAVE RESULTS
# =====================================================

results = {
    "model": "Wav2Vec2 + StandardScaler + PCA + SVM",
    "training_samples": int(len(y_train)),
    "testing_samples": int(len(y_test)),
    "embedding_features": int(len(embedding_columns)),
    "best_parameters": search.best_params_,
    "cv_macro_f1": float(search.best_score_),
    "training_accuracy": float(train_accuracy),
    "testing_accuracy": float(test_accuracy),
    "balanced_accuracy": float(balanced_acc),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1),
    "classification_report": classification_report(
        y_test,
        test_predictions,
        output_dict=True,
        zero_division=0
    ),
    "confusion_matrix": confusion_matrix(
        y_test,
        test_predictions,
        labels=sorted(set(y_train))
    ).tolist(),
    "class_labels": sorted(set(y_train))
}

with open(RESULT_OUTPUT, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4)

pd.DataFrame(search.cv_results_).to_csv(
    CV_OUTPUT,
    index=False
)

print("\nSaved model:", MODEL_OUTPUT)
print("Saved results:", RESULT_OUTPUT)
print("Saved CV results:", CV_OUTPUT)

print("\nExperiment completed successfully.")
