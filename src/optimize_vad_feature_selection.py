
import os
import json
import joblib
import pandas as pd

from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report
)

print("\nVAD FEATURE SELECTION + SVM OPTIMIZATION")
print("------------------------------------------")

# Load original 40-feature VAD datasets
train_df = pd.read_csv("features/train_features_vad.csv")
test_df = pd.read_csv("features/test_features_vad.csv")

metadata = ["filename", "emotion", "actor_id"]

X_train = train_df.drop(columns=metadata)
y_train = train_df["emotion"]

X_test = test_df[X_train.columns]
y_test = test_df["emotion"]

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Available features:", X_train.shape[1])

# Feature selection is performed inside each CV fold
def mutual_info_score(X, y):
    return mutual_info_classif(
        X, y, random_state=42
    )

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("selector", SelectKBest(score_func=mutual_info_score)),
    ("svm", SVC(kernel="rbf", class_weight="balanced"))
])

param_grid = {
    "selector__k": [10, 15, 20, 25, 30, 35, 40],
    "svm__C": [0.1, 1, 10, 100],
    "svm__gamma": ["scale", 0.01, 0.001]
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
    n_jobs=-1,
    verbose=1,
    refit=True
)

print("\nStarting 420 candidate combinations...")
search.fit(X_train, y_train)

best_model = search.best_estimator_

train_predictions = best_model.predict(X_train)
test_predictions = best_model.predict(X_test)

selected_mask = best_model.named_steps[
    "selector"
].get_support()

selected_features = list(
    X_train.columns[selected_mask]
)

results = {
    "best_parameters": {
        key: str(value)
        for key, value in search.best_params_.items()
    },
    "best_cv_macro_f1": float(search.best_score_),
    "selected_feature_count": len(selected_features),
    "selected_features": selected_features,
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

print("\nSELECTED FEATURES")
for feature in selected_features:
    print("-", feature)

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
        "selected_features": selected_features,
        "best_params": search.best_params_
    },
    "models/svm_ravdess_vad_feature_selected.pkl"
)

with open(
    "reports/vad_feature_selection_results.json",
    "w"
) as f:
    json.dump(results, f, indent=4)

pd.DataFrame(search.cv_results_).to_csv(
    "reports/vad_feature_selection_cv_results.csv",
    index=False
)

print("\nModel and reports saved successfully.")
