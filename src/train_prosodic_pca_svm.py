
import os
import json
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report
)

print("\nPROSODIC PCA + SVM TRAINING")
print("--------------------------------")

train_df = pd.read_csv(
    "features/train_features_vad_prosodic.csv"
)

test_df = pd.read_csv(
    "features/test_features_vad_prosodic.csv"
)

metadata = ["filename", "emotion", "actor_id"]

X_train = train_df.drop(columns=metadata)
y_train = train_df["emotion"]

X_test = test_df[X_train.columns]
y_test = test_df["emotion"]

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Features:", X_train.shape[1])

model = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=35)),
    ("svm", SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        class_weight="balanced"
    ))
])

print("\nTraining model...")
model.fit(X_train, y_train)

train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

results = {
    "model": "Prosodic_PCA_SVM",
    "features": X_train.shape[1],
    "training_samples": len(X_train),
    "testing_samples": len(X_test),
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
        y_test, test_predictions, output_dict=True
    )
}

print("\nRESULTS")
print("--------------------------------")
print(f"Training Accuracy: {results['training_accuracy']*100:.2f}%")
print(f"Testing Accuracy: {results['testing_accuracy']*100:.2f}%")
print(f"Balanced Accuracy: {results['balanced_accuracy']*100:.2f}%")
print(f"Macro F1: {results['macro_f1']*100:.2f}%")

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, test_predictions))

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

joblib.dump(
    {
        "model": model,
        "feature_columns": list(X_train.columns),
        "classes": list(model.classes_),
        "best_params": model.get_params()
    },
    "models/svm_ravdess_vad_prosodic_pca.pkl"
)

with open(
    "reports/prosodic_pca_svm_results.json", "w"
) as f:
    json.dump(results, f, indent=4)

print("\nModel saved successfully.")
print("Results saved successfully.")
