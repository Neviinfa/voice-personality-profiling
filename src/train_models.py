
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report
)

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]

train = pd.read_csv(ROOT / "features" / "train_features.csv")
test = pd.read_csv(ROOT / "features" / "test_features.csv")

metadata = ["filename", "emotion", "actor_id"]
feature_cols = [c for c in train.columns if c not in metadata]

X_train = train[feature_cols].astype(float)
X_test = test[feature_cols].astype(float)
y_train = train["emotion"]
y_test = test["emotion"]

if feature_cols != [c for c in test.columns if c not in metadata]:
    raise ValueError("Train/test feature columns do not match")

if not np.isfinite(X_train).all().all() or not np.isfinite(X_test).all().all():
    raise ValueError("Non-finite feature values detected")

# Encode labels for XGBoost
encoder = LabelEncoder()
y_train_enc = encoder.fit_transform(y_train)
y_test_enc = encoder.transform(y_test)

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

model_dir = ROOT / "models"
report_dir = ROOT / "reports"
model_dir.mkdir(exist_ok=True)
report_dir.mkdir(exist_ok=True)

results = []

for name, model in models.items():
    print(f"\nTraining {name}...")

    if name == "XGBoost":
        model.fit(X_train, y_train_enc)
        train_pred = encoder.inverse_transform(model.predict(X_train).astype(int))
        test_pred = encoder.inverse_transform(model.predict(X_test).astype(int))
    else:
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

    metrics = {
        "model": name,
        "training_accuracy": accuracy_score(y_train, train_pred),
        "testing_accuracy": accuracy_score(y_test, test_pred),
        "training_balanced_accuracy": balanced_accuracy_score(y_train, train_pred),
        "testing_balanced_accuracy": balanced_accuracy_score(y_test, test_pred),
        "training_macro_f1": f1_score(y_train, train_pred, average="macro"),
        "testing_macro_f1": f1_score(y_test, test_pred, average="macro")
    }

    results.append(metrics)

    joblib.dump(
        {"model": model, "features": feature_cols, "encoder": encoder if name == "XGBoost" else None},
        model_dir / f"{name.lower()}_ravdess.pkl"
    )

    print("\n", name, "RESULTS")
    for key, value in metrics.items():
        if key != "model":
            print(f"{key}: {value * 100:.2f}%")

    print("\nTest classification report:")
    print(classification_report(y_test, test_pred, zero_division=0))

comparison = pd.DataFrame(results)
comparison.to_csv(report_dir / "model_comparison.csv", index=False)

with open(report_dir / "model_comparison.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nFINAL MODEL COMPARISON")
print(comparison.to_string(index=False))
print("\nAll models and reports saved.")
