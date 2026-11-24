
from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
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

pipeline = Pipeline([
    ("selection", SelectKBest(score_func=mutual_info_classif)),
    ("scaler", StandardScaler()),
    ("classifier", SVC(
        kernel="rbf",
        class_weight="balanced"
    ))
])

param_grid = {
    "selection__k": [10, 20, 30, 40],
    "classifier__C": [0.1, 1, 10],
    "classifier__gamma": ["scale", 0.01]
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

print("Starting SVM feature selection...")

search.fit(X_train, y_train)

best_model = search.best_estimator_

train_pred = best_model.predict(X_train)
test_pred = best_model.predict(X_test)

results = {
    "best_parameters": search.best_params_,
    "best_cv_macro_f1": float(search.best_score_),
    "training_accuracy": float(accuracy_score(y_train, train_pred)),
    "testing_accuracy": float(accuracy_score(y_test, test_pred)),
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

selected_features = [
    feature_cols[i]
    for i in best_model.named_steps["selection"].get_support(indices=True)
]

model_dir = ROOT / "models"
report_dir = ROOT / "reports"

joblib.dump(
    {
        "model": best_model,
        "features": feature_cols,
        "selected_features": selected_features
    },
    model_dir / "svm_feature_selection_candidate.pkl"
)

with open(report_dir / "svm_feature_selection_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nBEST PARAMETERS:")
print(search.best_params_)

print("\nSELECTED FEATURES:")
print(selected_features)

print("\nFINAL RESULTS:")
for key, value in results.items():
    if isinstance(value, float):
        print(f"{key}: {value * 100:.2f}%")
    else:
        print(f"{key}: {value}")

print("\nTEST CLASSIFICATION REPORT:")
print(classification_report(y_test, test_pred, zero_division=0))

print("\nFEATURE SELECTION COMPLETED")
