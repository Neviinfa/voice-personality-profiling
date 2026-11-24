
import os
import json
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report
)

# Project paths
TRAIN_PATH = "features/train_features_vad.csv"
TEST_PATH = "features/test_features_vad.csv"

# Load datasets
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# Separate features and labels
metadata = ["filename", "emotion", "actor_id"]

X_train = train_df.drop(columns=metadata, errors="ignore")
X_test = test_df.drop(columns=metadata, errors="ignore")

y_train = train_df["emotion"]
y_test = test_df["emotion"]

# Validate feature compatibility
if list(X_train.columns) != list(X_test.columns):
    raise ValueError("Training and testing features do not match!")

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)
print("\nStarting Random Forest optimization...")

# Random Forest model
rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)

# Hyperparameter search
param_grid = {
    "n_estimators": [300],
    "max_depth": [None, 20, 40],
    "max_features": ["sqrt", "log2", 0.5],
    "min_samples_leaf": [1, 2],
    "class_weight": [None, "balanced_subsample"]
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

grid = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=2
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# Predictions
train_predictions = best_model.predict(X_train)
test_predictions = best_model.predict(X_test)

# Metrics
results = {
    "best_parameters": grid.best_params_,
    "cv_macro_f1": float(grid.best_score_),
    "train_accuracy": float(
        accuracy_score(y_train, train_predictions)
    ),
    "test_accuracy": float(
        accuracy_score(y_test, test_predictions)
    ),
    "test_balanced_accuracy": float(
        balanced_accuracy_score(y_test, test_predictions)
    ),
    "test_macro_f1": float(
        f1_score(y_test, test_predictions, average="macro")
    )
}

print("\n========== RANDOM FOREST RESULTS ==========")
print("Best parameters:", grid.best_params_)
print(f"CV Macro F1: {results['cv_macro_f1']:.4f}")
print(f"Training Accuracy: {results['train_accuracy'] * 100:.2f}%")
print(f"Testing Accuracy: {results['test_accuracy'] * 100:.2f}%")
print(
    f"Balanced Accuracy: "
    f"{results['test_balanced_accuracy'] * 100:.2f}%"
)
print(f"Macro F1: {results['test_macro_f1']:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, test_predictions))

# Save separately
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

joblib.dump(
    {
        "model": best_model,
        "feature_columns": list(X_train.columns)
    },
    "models/random_forest_ravdess_optimized.pkl"
)

with open(
    "reports/random_forest_optimization_results.json",
    "w"
) as f:
    json.dump(results, f, indent=4)

print("\nModel and results saved successfully.")
