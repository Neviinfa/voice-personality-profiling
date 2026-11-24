
import os
import json
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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

train_df = pd.read_csv(
    os.path.join(FEATURE_DIR, "train_features_vad.csv")
)

test_df = pd.read_csv(
    os.path.join(FEATURE_DIR, "test_features_vad.csv")
)

metadata = ["filename", "emotion", "actor_id"]
features = [c for c in train_df.columns if c not in metadata]

assert features == [c for c in test_df.columns if c not in metadata]

X_train = train_df[features]
X_test = test_df[features]
y_train = train_df["emotion"]
y_test = test_df["emotion"]

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(class_weight="balanced"))
])

param_grid = [
    {
        "svm__kernel": ["linear"],
        "svm__C": [0.01, 0.1, 1, 10, 100]
    },
    {
        "svm__kernel": ["rbf"],
        "svm__C": [0.1, 1, 10, 100],
        "svm__gamma": ["scale", 0.01, 0.001]
    },
    {
        "svm__kernel": ["poly"],
        "svm__C": [0.1, 1, 10],
        "svm__gamma": ["scale", 0.01],
        "svm__degree": [2, 3]
    }
]

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
    n_jobs=4,
    verbose=2,
    refit=True
)

print("\nStarting SVM kernel optimization...")
print("Training samples:", len(X_train))
print("Features:", len(features))
print("Cross-validation: 5-fold stratified")

search.fit(X_train, y_train)

best_model = search.best_estimator_

print("\nBEST PARAMETERS")
print(search.best_params_)
print(
    "Best CV Macro-F1:",
    round(search.best_score_ * 100, 2), "%"
)

# Evaluate the selected model on the test set
train_pred = best_model.predict(X_train)
test_pred = best_model.predict(X_test)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)
balanced_acc = balanced_accuracy_score(y_test, test_pred)
macro_f1 = f1_score(y_test, test_pred, average="macro")

print("\nFINAL SVM RESULTS")
print("-------------------------")
print(f"Training accuracy: {train_acc * 100:.2f}%")
print(f"Testing accuracy: {test_acc * 100:.2f}%")
print(f"Balanced accuracy: {balanced_acc * 100:.2f}%")
print(f"Macro F1: {macro_f1 * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, test_pred, zero_division=0))

# Save as a separate candidate model
joblib.dump(
    {
        "model": best_model,
        "feature_columns": features,
        "classes": list(best_model.classes_),
        "best_params": search.best_params_
    },
    os.path.join(MODEL_DIR, "svm_ravdess_vad_optimized.pkl")
)

results = {
    "best_params": search.best_params_,
    "best_cv_macro_f1": float(search.best_score_),
    "train_accuracy": float(train_acc),
    "test_accuracy": float(test_acc),
    "balanced_accuracy": float(balanced_acc),
    "macro_f1": float(macro_f1)
}

with open(
    os.path.join(REPORT_DIR, "svm_optimization_results.json"),
    "w"
) as f:
    json.dump(results, f, indent=4)

pd.DataFrame(search.cv_results_).to_csv(
    os.path.join(REPORT_DIR, "svm_kernel_cv_results.csv"),
    index=False
)

print("\nOptimized model and reports saved successfully!")
