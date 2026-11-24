
import os
import json
import joblib
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report
)

# Load datasets
train_df = pd.read_csv("features/train_features_vad.csv")
test_df = pd.read_csv("features/test_features_vad.csv")

metadata = ["filename", "emotion", "actor_id"]

X_train = train_df.drop(columns=metadata, errors="ignore")
X_test = test_df.drop(columns=metadata, errors="ignore")

y_train_text = train_df["emotion"]
y_test_text = test_df["emotion"]

if list(X_train.columns) != list(X_test.columns):
    raise ValueError("Training and testing features do not match!")

# Encode labels
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train_text)
y_test = encoder.transform(y_test_text)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)
print("Classes:", list(encoder.classes_))
print("\nStarting XGBoost optimization...")

# Base model
xgb = XGBClassifier(
    objective="multi:softprob",
    eval_metric="mlogloss",
    tree_method="hist",
    random_state=42,
    n_jobs=1
)

# Hyperparameter grid: 64 combinations
param_grid = {
    "n_estimators": [300, 500],
    "max_depth": [2, 3],
    "learning_rate": [0.03, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "reg_lambda": [1, 5]
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

grid = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=2,
    refit=True
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

print("\n========== XGBOOST RESULTS ==========")
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
print(
    classification_report(
        y_test_text,
        encoder.inverse_transform(test_predictions),
        labels=encoder.classes_
    )
)

# Save model and reports separately
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

joblib.dump(
    {
        "model": best_model,
        "label_encoder": encoder,
        "feature_columns": list(X_train.columns)
    },
    "models/xgboost_ravdess_optimized.pkl"
)

with open(
    "reports/xgboost_optimization_results.json",
    "w"
) as f:
    json.dump(results, f, indent=4)

pd.DataFrame(grid.cv_results_).to_csv(
    "reports/xgboost_optimization_cv.csv",
    index=False
)

print("\nModel and reports saved successfully.")
