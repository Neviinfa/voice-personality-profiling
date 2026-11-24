
import os
import json
import joblib
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

print("\nADVANCED SVM KERNEL OPTIMIZATION")
print("--------------------------------")

# Load original VAD features
train_df = pd.read_csv("features/train_features_vad.csv")
test_df = pd.read_csv("features/test_features_vad.csv")

metadata = ["filename", "emotion", "actor_id"]

X_train = train_df.drop(columns=metadata)
y_train = train_df["emotion"]

X_test = test_df[X_train.columns]
y_test = test_df["emotion"]

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Features:", X_train.shape[1])

# All preprocessing stays inside the CV pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA()),
    ("svm", SVC(class_weight="balanced"))
])

# Separate grids for each kernel
param_grid = [
    {
        "pca__n_components": [25, 30, 35, 40],
        "svm__kernel": ["rbf"],
        "svm__C": [1, 10, 100],
        "svm__gamma": ["scale", 0.01, 0.001]
    },
    {
        "pca__n_components": [25, 30, 35, 40],
        "svm__kernel": ["poly"],
        "svm__C": [1, 10, 100],
        "svm__gamma": ["scale", 0.01],
        "svm__degree": [2, 3]
    },
    {
        "pca__n_components": [25, 30, 35, 40],
        "svm__kernel": ["sigmoid"],
        "svm__C": [1, 10, 100],
        "svm__gamma": ["scale", 0.01, 0.001]
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
    n_jobs=-1,
    verbose=2,
    refit=True
)

print("\nStarting advanced kernel optimization...")

search.fit(X_train, y_train)

best_model = search.best_estimator_

# Evaluate the selected model
train_predictions = best_model.predict(X_train)
test_predictions = best_model.predict(X_test)

results = {
    "best_parameters": {
        key: str(value)
        for key, value in search.best_params_.items()
    },
    "best_cv_macro_f1": float(search.best_score_),
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

print("\nBEST PARAMETERS")
print(search.best_params_)

print("\nBEST CV MACRO-F1:",
      f"{search.best_score_ * 100:.2f}%")

print("\nTRAINING ACCURACY:",
      f"{results['training_accuracy'] * 100:.2f}%")

print("TESTING ACCURACY:",
      f"{results['testing_accuracy'] * 100:.2f}%")

print("BALANCED ACCURACY:",
      f"{results['balanced_accuracy'] * 100:.2f}%")

print("MACRO-F1:",
      f"{results['macro_f1'] * 100:.2f}%")

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, test_predictions))

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

joblib.dump(
    {
        "model": best_model,
        "feature_columns": list(X_train.columns),
        "classes": list(best_model.classes_),
        "best_params": search.best_params_
    },
    "models/svm_ravdess_advanced_kernel.pkl"
)

with open(
    "reports/advanced_svm_kernel_results.json",
    "w"
) as f:
    json.dump(results, f, indent=4)

pd.DataFrame(search.cv_results_).to_csv(
    "reports/advanced_svm_kernel_cv.csv",
    index=False
)

print("\nModel saved successfully.")
print("Results saved successfully.")
