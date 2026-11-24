
from pathlib import Path
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

ROOT = Path(__file__).resolve().parents[1]

train = pd.read_csv(ROOT / "features" / "train_features.csv")
test = pd.read_csv(ROOT / "features" / "test_features.csv")

metadata = ["filename", "emotion", "actor_id"]
feature_cols = [c for c in train.columns if c not in metadata]

X_train = train[feature_cols].astype(float)
X_test = test[feature_cols].astype(float)
y_train = train["emotion"]
y_test = test["emotion"]

if feature_cols != [
    c for c in test.columns if c not in metadata
]:
    raise ValueError("Feature columns do not match")

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", SVC(
        kernel="rbf",
        class_weight="balanced"
    ))
])

param_grid = {
    "classifier__C": [0.01, 0.1, 1, 3, 10],
    "classifier__gamma": ["scale", 0.01, 0.001]
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
    verbose=2,
    refit=True
)

print("Starting SVM hyperparameter tuning...")

search.fit(X_train, y_train)

best_model = search.best_estimator_

train_pred = best_model.predict(X_train)
test_pred = best_model.predict(X_test)

results = {
    "best_parameters": search.best_params_,
    "best_cv_macro_f1": float(search.best_score_),
    "training_accuracy": float(
        accuracy_score(y_train, train_pred)
    ),
    "testing_accuracy": float(
        accuracy_score(y_test, test_pred)
    ),
    "training_balanced_accuracy": float(
        balanced_accuracy_score(y_train, train_pred)
    ),
    "testing_balanced_accuracy": float(
        balanced_accuracy_score(y_test, test_pred)
    ),
    "training_macro_f1": float(
        f1_score(y_train, train_pred, average="macro")
    ),
    "testing_macro_f1": float(
        f1_score(y_test, test_pred, average="macro")
    )
}

model_dir = ROOT / "models"
report_dir = ROOT / "reports"

model_dir.mkdir(exist_ok=True)
report_dir.mkdir(exist_ok=True)

joblib.dump(
    {
        "model": best_model,
        "features": feature_cols
    },
    model_dir / "svm_ravdess_tuned_candidate.pkl"
)

with open(
    report_dir / "svm_tuning_results.json", "w"
) as f:
    json.dump(results, f, indent=4)

print("\nBEST PARAMETERS:")
print(search.best_params_)

print("\nFINAL SVM RESULTS:")
for key, value in results.items():
    if isinstance(value, float):
        print(f"{key}: {value * 100:.2f}%")
    else:
        print(f"{key}: {value}")

print("\nTEST CLASSIFICATION REPORT:")
print(classification_report(
    y_test,
    test_pred,
    zero_division=0
))

print("\nTUNED SVM CANDIDATE SAVED SUCCESSFULLY")
