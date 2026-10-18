"""
STEP 1 - Train the attack classifier.

Reads UNSW-NB15, trains a Random Forest that labels each network flow as
Normal or one of 9 attack types, measures how good it is, and saves
the model plus a report for the dashboard.

Run:  python step1_train_model.py
"""

import json
import time
import warnings

import numpy as np

import joblib
import matplotlib
matplotlib.use("Agg")                 # draw charts to files, no window needed
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from common import ARTIFACTS, RANDOM_STATE, TARGET, load_splits, make_encoder, to_features

# 1. Load data -------------------------------------------------------------
print("Loading UNSW-NB15 ...")
train_df, test_df = load_splits()
print(f"  train rows: {len(train_df):,}   test rows: {len(test_df):,}")

# 2. Prepare features --------------------------------------------------------
encoder = make_encoder(train_df)
X_train, y_train = to_features(train_df, encoder), train_df[TARGET]
X_test, y_test = to_features(test_df, encoder), test_df[TARGET]
feature_cols = list(X_train.columns)
print(f"  using {len(feature_cols)} features")



def score(y_true, y_pred):
    """The four numbers we report for every model."""
    attack = y_true != "Normal"
    flagged = y_pred != "Normal"
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "detection_rate": float((flagged & attack).sum() / attack.sum()),
        "false_alarm_rate": float((flagged & ~attack).sum() / (~attack).sum()),
    }


# 3. Compare other algorithms (justifies why we chose Random Forest) -------
warnings.filterwarnings("ignore")        # hide convergence chatter from LR
candidates = {
    # log1p squashes huge byte counts, scaling puts features on equal footing
    "Logistic Regression": make_pipeline(FunctionTransformer(np.log1p), StandardScaler(),
                                         LogisticRegression(max_iter=300)),
    "Decision Tree": DecisionTreeClassifier(max_depth=14, min_samples_leaf=5,
                                            random_state=RANDOM_STATE),
    "Gradient Boosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
}
comparison = {}
for name, candidate in candidates.items():
    print(f"Comparing: {name} ...")
    start = time.time()
    candidate.fit(X_train, y_train)
    comparison[name] = score(y_test, candidate.predict(X_test))
    comparison[name]["train_seconds"] = round(time.time() - start, 1)

# 4. Train the model we actually use ----------------------------------------
# 60 trees, each limited to depth 14. Deeper/bigger forests were only ~0.3%
# more accurate but ~10x slower to explain with SHAP in step 3.
model = RandomForestClassifier(
    n_estimators=60,
    max_depth=14,
    min_samples_leaf=5,
    n_jobs=-1,                 # use every CPU core
    random_state=RANDOM_STATE,
)
print("Training Random Forest (under a minute on most laptops) ...")
start = time.time()
model.fit(X_train, y_train)
train_seconds = time.time() - start
print(f"  done in {train_seconds:.0f}s")

# 5. Evaluate --------------------------------------------------------------
y_pred = model.predict(X_test)

# Multi-class: did we get the exact attack type right?
accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")

# Binary: did we at least notice it was an attack? (what matters most for alerts)
actual_attack = y_test != "Normal"
flagged = y_pred != "Normal"
detection_rate = (flagged & actual_attack).sum() / actual_attack.sum()
false_alarm_rate = (flagged & ~actual_attack).sum() / (~actual_attack).sum()

print(f"\nAttack-type accuracy : {accuracy:.3f}")
print(f"Macro F1             : {macro_f1:.3f}")
print(f"Detection rate       : {detection_rate:.3f}  (attacks caught)")
print(f"False alarm rate     : {false_alarm_rate:.3f}  (normal traffic wrongly flagged)\n")
report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
print(classification_report(y_test, y_pred, zero_division=0))

comparison["Random Forest (chosen)"] = score(y_test, y_pred)
comparison["Random Forest (chosen)"]["train_seconds"] = round(train_seconds, 1)
print("Model comparison:")
print(pd.DataFrame(comparison).T.round(3).to_string())
print()

# 6. Save everything -------------------------------------------------------
joblib.dump(model, ARTIFACTS / "model.joblib", compress=3)
joblib.dump(encoder, ARTIFACTS / "encoder.joblib")
test_df.to_csv(ARTIFACTS / "test_split.csv", index=False)

metrics = {
    "accuracy": accuracy,
    "macro_f1": macro_f1,
    "detection_rate": float(detection_rate),
    "false_alarm_rate": float(false_alarm_rate),
    "train_rows": len(train_df),
    "test_rows": len(test_df),
    "feature_cols": feature_cols,
    "classes": list(model.classes_),
    "per_class": {c: report[c] for c in model.classes_ if c in report},
    "comparison": comparison,
}
with open(ARTIFACTS / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# Confusion matrix picture: rows = true type, columns = predicted type
fig, ax = plt.subplots(figsize=(9, 8))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, normalize="true", values_format=".2f",
    cmap="Blues", ax=ax, xticks_rotation=45, colorbar=False,
)
ax.set_title("Confusion matrix (each row sums to 1)")
fig.tight_layout()
fig.savefig(ARTIFACTS / "confusion_matrix.png", dpi=130)

print(f"Saved model, encoder, metrics and confusion matrix to {ARTIFACTS}/")
