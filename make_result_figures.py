"""
make_result_figures.py

Generates the figures used in the Review 2 report from the trained model and
the held-out test split:
  - roc_curve.png      : ROC curve (attack vs normal) with AUC
  - pr_curve.png       : precision-recall curve with average precision
  - model_comparison.png : bar chart of the four benchmarked models
All are plain black/white-friendly matplotlib figures.

Run after run_pipeline.py:  python make_result_figures.py
"""

import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             average_precision_score)

from common import ARTIFACTS, TARGET, to_features

model = joblib.load(ARTIFACTS / "model.joblib")
encoder = joblib.load(ARTIFACTS / "encoder.joblib")
meta = json.loads((ARTIFACTS / "metrics.json").read_text())
cols = meta["feature_cols"]

test = pd.read_csv(ARTIFACTS / "test_split.csv")
X = to_features(test, encoder)[cols]
y_true_attack = (test[TARGET] != "Normal").astype(int).to_numpy()

proba = model.predict_proba(X)
normal_idx = list(model.classes_).index("Normal")
attack_score = 1 - proba[:, normal_idx]

# --- ROC curve ---
fpr, tpr, _ = roc_curve(y_true_attack, attack_score)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(5.2, 4.4))
plt.plot(fpr, tpr, color="black", lw=2, label=f"Random Forest (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--", label="Random guess")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate (detection rate)")
plt.title("ROC curve: attack vs normal")
plt.legend(loc="lower right")
plt.grid(True, linestyle=":", alpha=0.5)
plt.tight_layout()
plt.savefig(ARTIFACTS / "roc_curve.png", dpi=150)
plt.close()

# --- Precision-recall curve ---
prec, rec, _ = precision_recall_curve(y_true_attack, attack_score)
ap = average_precision_score(y_true_attack, attack_score)
plt.figure(figsize=(5.2, 4.4))
plt.plot(rec, prec, color="black", lw=2, label=f"Random Forest (AP = {ap:.3f})")
plt.xlabel("Recall (detection rate)")
plt.ylabel("Precision")
plt.title("Precision-recall curve: attack vs normal")
plt.legend(loc="lower left")
plt.grid(True, linestyle=":", alpha=0.5)
plt.tight_layout()
plt.savefig(ARTIFACTS / "pr_curve.png", dpi=150)
plt.close()

# --- Model comparison bar chart ---
comp = meta["comparison"]
models = list(comp.keys())
detection = [comp[m]["detection_rate"] * 100 for m in models]
false_alarm = [comp[m]["false_alarm_rate"] * 100 for m in models]
short = [m.replace(" (chosen)", "*") for m in models]

x = np.arange(len(models))
w = 0.38
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.bar(x - w/2, detection, w, label="Detection rate", color="black")
ax.bar(x + w/2, false_alarm, w, label="False-alarm rate", color="grey", hatch="///")
ax.set_ylabel("Percent")
ax.set_title("Model comparison (held-out test set)")
ax.set_xticks(x)
ax.set_xticklabels(short, rotation=15, ha="right")
ax.legend()
ax.grid(True, axis="y", linestyle=":", alpha=0.5)
for i, (d, f) in enumerate(zip(detection, false_alarm)):
    ax.text(i - w/2, d + 1, f"{d:.0f}", ha="center", fontsize=8)
    ax.text(i + w/2, f + 1, f"{f:.1f}", ha="center", fontsize=8)
plt.tight_layout()
plt.savefig(ARTIFACTS / "model_comparison.png", dpi=150)
plt.close()

summary = {"roc_auc": round(roc_auc, 4), "average_precision": round(ap, 4)}
(ARTIFACTS / "curve_metrics.json").write_text(json.dumps(summary, indent=2))
print(f"ROC AUC = {roc_auc:.4f},  Average precision = {ap:.4f}")
print("Saved roc_curve.png, pr_curve.png, model_comparison.png")
