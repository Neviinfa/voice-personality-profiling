
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

print("\nSVM CONFUSION MATRIX AND ERROR ANALYSIS")
print("----------------------------------------")

# Load best PCA + SVM model
MODEL_PATH = "models/svm_ravdess_vad_pca.pkl"

saved_data = joblib.load(MODEL_PATH)

model = saved_data["model"]
feature_columns = saved_data["feature_columns"]

# Load original test dataset
test_df = pd.read_csv(
    "features/test_features_vad.csv"
)

metadata = ["filename", "emotion", "actor_id"]

X_test = test_df[feature_columns]
y_test = test_df["emotion"]

# Predict
predictions = model.predict(X_test)

print("\nTesting Accuracy:",
      f"{accuracy_score(y_test, predictions) * 100:.2f}%")

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, predictions))

# Confusion matrix
labels = sorted(y_test.unique())

cm = confusion_matrix(
    y_test,
    predictions,
    labels=labels
)

os.makedirs("reports", exist_ok=True)

plt.figure(figsize=(11, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.title("RAVDESS PCA + SVM Confusion Matrix")
plt.xlabel("Predicted Emotion")
plt.ylabel("Actual Emotion")
plt.tight_layout()

plt.savefig(
    "reports/svm_confusion_matrix.png",
    dpi=300
)

plt.close()

# Identify incorrect predictions
results = test_df[
    ["filename", "emotion", "actor_id"]
].copy()

results["predicted_emotion"] = predictions

results["correct"] = (
    results["emotion"] == results["predicted_emotion"]
)

misclassified = results[
    results["correct"] == False
].copy()

misclassified.to_csv(
    "reports/svm_misclassified_recordings.csv",
    index=False
)

# Emotion-wise error counts
error_summary = (
    misclassified.groupby(
        ["emotion", "predicted_emotion"]
    )
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

error_summary.to_csv(
    "reports/svm_emotion_errors.csv",
    index=False
)

print("\nTOP CONFUSED EMOTION PAIRS")
print(error_summary.head(15).to_string(index=False))

print("\nTOTAL TEST RECORDINGS:", len(test_df))
print("CORRECT PREDICTIONS:", results["correct"].sum())
print("INCORRECT PREDICTIONS:", len(misclassified))

print("\nSAVED FILES:")
print("reports/svm_confusion_matrix.png")
print("reports/svm_misclassified_recordings.csv")
print("reports/svm_emotion_errors.csv")

print("\nERROR ANALYSIS COMPLETED!")
