"""
live_classifier.py  (Feature 1 - live attack-type classifier)

Live mode detects THAT a window is abnormal using the behavioural baseline.
This module decides WHICH attack it is. Earlier the type came from a few
hand-written if/else rules (label_window); here a small Decision Tree learns
the mapping from the ten live traffic features to an attack class.

Why a Decision Tree? It is small, fast, and - like the rest of this project -
interpretable: its decisions are readable if/else paths, so naming an attack
never becomes a black box.

Training data is generated here, because no public dataset uses this exact
10-feature live schema. Each class is synthesised from the shape its traffic
has in reality (a scan touches many ports with many SYNs; a flood sends a
huge number of inbound packets; brute force rides on failed logins with
little traffic; normal traffic stays small). Gaussian noise is added so the
tree sees variety, and the classes are deliberately separable on the same
features the baseline already computes.

Run:  python live_classifier.py        (trains and saves the model)
It is also imported by live_monitor.py, which calls load() and classify().
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from common import ARTIFACTS

# The ten live traffic features, in the exact order live_monitor computes them.
# Defined here (not imported from live_monitor) to avoid a circular import.
FEATURES = ["pkts_in", "pkts_out", "bytes_in", "bytes_out", "local_ports",
            "remote_ports", "syn_in", "rst_out", "icmp_in", "avg_pkt_size"]

LIVE_DIR = ARTIFACTS / "live"
LIVE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_FILE = LIVE_DIR / "attack_classifier.joblib"
REPORT_FILE = LIVE_DIR / "classifier_report.json"
RANDOM_STATE = 42
PER_CLASS = 1500          # training rows generated per class

CLASSES = ["Normal", "Port scan", "Traffic flood", "Brute force"]

# FEATURES order (from live_monitor):
# pkts_in, pkts_out, bytes_in, bytes_out, local_ports,
# remote_ports, syn_in, rst_out, icmp_in, avg_pkt_size


def _sample(n, spec, rng):
    """Draw n rows; spec maps feature -> (low, high) uniform range."""
    rows = {}
    for f in FEATURES:
        lo, hi = spec.get(f, (0, 1))
        rows[f] = rng.uniform(lo, hi, n)
    return pd.DataFrame(rows)[FEATURES]


def make_training_data():
    """Synthesise labelled live-traffic windows for each attack class."""
    rng = np.random.default_rng(RANDOM_STATE)
    frames = []

    # Normal browsing: a few ports, modest packets, balanced in/out.
    normal = _sample(PER_CLASS, {
        "pkts_in": (0, 60), "pkts_out": (0, 60), "bytes_in": (0, 8000), "bytes_out": (0, 6000),
        "local_ports": (1, 3), "remote_ports": (1, 4), "syn_in": (0, 3), "rst_out": (0, 2),
        "icmp_in": (0, 1), "avg_pkt_size": (60, 1400),
    }, rng)
    normal["label"] = "Normal"
    frames.append(normal)

    # Port scan: many of OUR ports touched, many SYNs, many refused (RST),
    # small packets, little data.
    scan = _sample(PER_CLASS, {
        "pkts_in": (40, 400), "pkts_out": (0, 60), "bytes_in": (2000, 30000), "bytes_out": (0, 4000),
        "local_ports": (15, 1024), "remote_ports": (1, 5), "syn_in": (40, 500), "rst_out": (10, 400),
        "icmp_in": (0, 2), "avg_pkt_size": (40, 120),
    }, rng)
    scan["label"] = "Port scan"
    frames.append(scan)

    # Traffic flood: huge inbound packet count on ONE port, lots of bytes.
    flood = _sample(PER_CLASS, {
        "pkts_in": (300, 6000), "pkts_out": (0, 50), "bytes_in": (20000, 800000), "bytes_out": (0, 3000),
        "local_ports": (1, 2), "remote_ports": (1, 2), "syn_in": (0, 50), "rst_out": (0, 10),
        "icmp_in": (0, 1500), "avg_pkt_size": (40, 200),
    }, rng)
    flood["label"] = "Traffic flood"
    frames.append(flood)

    # Brute force: little or no traffic; the signal is in failed logins, which
    # the monitor adds separately. We train the traffic part as near-silent so
    # the classifier does not confuse it with a scan or flood.
    brute = _sample(PER_CLASS, {
        "pkts_in": (0, 40), "pkts_out": (0, 40), "bytes_in": (0, 4000), "bytes_out": (0, 4000),
        "local_ports": (1, 3), "remote_ports": (1, 2), "syn_in": (0, 10), "rst_out": (0, 3),
        "icmp_in": (0, 1), "avg_pkt_size": (60, 600),
    }, rng)
    brute["label"] = "Brute force"
    frames.append(brute)

    data = pd.concat(frames, ignore_index=True)
    return data


def train():
    data = make_training_data()
    X, y = data[FEATURES], data["label"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE)

    clf = DecisionTreeClassifier(max_depth=8, min_samples_leaf=20, random_state=RANDOM_STATE)
    clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    report = classification_report(y_te, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_te, y_pred, labels=clf.classes_).tolist()

    joblib.dump(clf, MODEL_FILE)
    with open(REPORT_FILE, "w") as f:
        json.dump({"accuracy": acc, "classes": list(clf.classes_),
                   "per_class": {c: report[c] for c in clf.classes_},
                   "confusion_matrix": cm, "test_rows": len(y_te)}, f, indent=2)

    print(f"Live attack-type classifier trained on {len(X_tr):,} windows.")
    print(f"  test accuracy: {acc:.3f}  (on {len(y_te):,} held-out windows)")
    print(classification_report(y_te, y_pred, zero_division=0))
    print(f"Saved to {MODEL_FILE}")
    return clf


def load():
    """Return the trained classifier, or None if it was never trained."""
    return joblib.load(MODEL_FILE) if MODEL_FILE.exists() else None


def classify(clf, feature_row, failed_logins):
    """
    Name the attack for one window.

    feature_row: a pandas Series or dict with the FEATURES keys.
    failed_logins: failed-login count for this IP (host evidence the
    traffic classifier cannot see). Brute force is asserted here when the
    login signal is strong, matching the correlation rule in the monitor.
    """
    if failed_logins >= 5:
        return "Brute force"
    if clf is None:
        return None
    x = pd.DataFrame([[float(feature_row[f]) for f in FEATURES]], columns=FEATURES)
    pred = clf.predict(x)[0]
    # The model can call a near-silent window "Brute force" from traffic alone;
    # without the login signal that is really just unusual traffic.
    if pred == "Brute force":
        return "Unusual traffic"
    return pred


if __name__ == "__main__":
    train()
