
import json
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

print("\nCLASS-WEIGHT OPTIMIZATION: PCA + SVM")
print("------------------------------------")

# Load the existing VAD feature datasets
train_df = pd.read_csv("features/train_features_vad.csv")
test_df = pd.read_csv("features/test_features_vad.csv")

metadata = ["filename", "emotion", "actor_id"]

X_train = train_df.drop(columns=metadata)
y_train = train_df["emotion"]

X_test = test_df.drop(columns=metadata)
y_test = test_df["emotion"]

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Features:", X_train.shape[1])

# PCA + SVM pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=35)),
    ("svm", SVC(
        kernel="rbf",
        C=10,
        gamma="scale"
    ))
])

# Compare different class-weight strategies
class_weights = [
    None,
    "balanced",
    {
        "angry": 1.0,
        "calm": 1.0,
        "disgust": 1.0,
        "fearful": 1.0,
        "happy": 1.3,
        "neutral": 1.3,
        "sad": 1.3,
        "surprised": 1.0
    },
    {
        "angry": 1.0,
        "calm": 1.0,
        "disgust": 1.0,
        "fearful": 1.0,
        "happy": 1.5,
        "neutral": 1.5,
        "sad": 1.5,
        "surprised": 1.0
    }
]

param_grid = {
    "svm__class_weight": class_weights
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
    verbose=2
)

print("\nStarting class-weight search...")
search.fit(X_train, y_train)

best_model = search.best_estimator_

train_predictions = best_model.predict(X_train)
test_predictions = best_model.predict(X_test)

results = {
    "best_parameters": str(search.best_params_),
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

print("\nBEST PARAMETERS:")
print(search.best_params_)

print("\nTRAINING ACCURACY:",
      f"{results['training_accuracy'] * 100:.2f}%")

print("TESTING ACCURACY:",
      f"{results['testing_accuracy'] * 100:.2f}%")

print("BALANCED ACCURACY:",
      f"{results['balanced_accuracy'] * 100:.2f}%")

print("MACRO F1:",
      f"{results['macro_f1'] * 100:.2f}%")

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, test_predictions))

# Save as a separate candidate model
import joblib

joblib.dump(
    best_model,
    "models/svm_ravdess_vad_pca_classweight.pkl"
)

with open(
    "reports/class_weight_pca_svm_results.json",
    "w"
) as f:
    json.dump(results, f, indent=4)

pd.DataFrame(search.cv_results_).to_csv(
    "reports/class_weight_pca_svm_cv_results.csv",
    index=False
)

print("\nModel and results saved successfully.")
