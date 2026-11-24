
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

print("\nAUGMENTED PCA + SVM TRAINING")
print("--------------------------------")

train_df = pd.read_csv(
    "features/train_features_vad_augmented.csv"
)

test_df = pd.read_csv(
    "features/test_features_vad.csv"
)

metadata = ["filename", "emotion", "actor_id"]

X_train = train_df.drop(columns=metadata)
y_train = train_df["emotion"]

X_test = test_df.drop(columns=metadata)
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
    "training_samples": len(X_train),
    "testing_samples": len(X_test),
    "features": X_train.shape[1],
    "pca_components": 35,
    "svm_C": 10,
    "svm_gamma": "scale",
    "class_weight": "balanced",
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

print("\nRESULTS")
print("--------------------------------")

print(
    "Training Accuracy:",
    f"{results['training_accuracy'] * 100:.2f}%"
)

print(
    "Testing Accuracy:",
    f"{results['testing_accuracy'] * 100:.2f}%"
)

print(
    "Balanced Accuracy:",
    f"{results['balanced_accuracy'] * 100:.2f}%"
)

print(
    "Macro F1:",
    f"{results['macro_f1'] * 100:.2f}%"
)

print("\nCLASSIFICATION REPORT")
print(
    classification_report(
        y_test,
        test_predictions
    )
)

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

joblib.dump(
    model,
    "models/svm_ravdess_augmented_pca.pkl"
)

with open(
    "reports/augmented_pca_svm_results.json",
    "w"
) as file:
    json.dump(results, file, indent=4)

print("\nModel saved successfully.")
print("Results saved successfully.")
