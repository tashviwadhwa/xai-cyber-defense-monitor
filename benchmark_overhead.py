"""
benchmark_overhead.py

Measures the performance cost of the security/explainability layer, for the
Review 2 "performance overhead of the security component" requirement.

It separates two things:
  (a) the raw detector - the model's prediction only, and
  (b) the full security pipeline - prediction + SHAP explanation
      (replay) or deviation explanation (live) + risk correlation.

The difference is the overhead the explainability and correlation add on top
of plain detection. Results are printed and saved to artifacts/overhead.json.

Run:  python benchmark_overhead.py
"""

import json
import time

import joblib
import numpy as np
import pandas as pd
import shap

from common import ARTIFACTS, to_features
import live_classifier

N_REPEAT = 5


def time_it(fn, n=N_REPEAT):
    """Median wall-clock milliseconds over n runs."""
    times = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000)
    return float(np.median(times))


def bench_replay():
    model = joblib.load(ARTIFACTS / "model.joblib")
    encoder = joblib.load(ARTIFACTS / "encoder.joblib")
    meta = json.loads((ARTIFACTS / "metrics.json").read_text())
    cols = meta["feature_cols"]
    feed = pd.read_csv(ARTIFACTS / "feed_flows.csv")
    X = to_features(feed, encoder)[cols]
    n = len(X)

    # (a) detection only: predict_proba for all rows
    detect_ms = time_it(lambda: model.predict_proba(X), n=3)

    # (b) explanation: TreeSHAP per alert. Timed on a fixed 500-alert sample
    # (one repeat) because the per-alert figure is what matters and is stable.
    explainer = shap.TreeExplainer(model)
    proba = model.predict_proba(X)
    normal_idx = list(model.classes_).index("Normal")
    attack_prob = 1 - proba[:, normal_idx]
    alert_rows = np.where(attack_prob >= 0.40)[0]
    sample = alert_rows[:500]
    Xa = X.iloc[sample]
    shap_ms = time_it(lambda: explainer.shap_values(Xa), n=1)

    per_flow_detect = detect_ms / n
    per_alert_shap = shap_ms / max(len(sample), 1)
    return {
        "flows_scored": n,
        "alerts_explained": int(len(alert_rows)),
        "shap_sample_size": int(len(sample)),
        "detect_total_ms": round(detect_ms, 1),
        "detect_per_flow_ms": round(per_flow_detect, 4),
        "shap_per_alert_ms": round(per_alert_shap, 2),
        "overhead_factor_per_alert": round(per_alert_shap / per_flow_detect, 1),
    }


def bench_live():
    clf = live_classifier.load()
    if clf is None:
        return {"note": "classifier not trained; run live_classifier.py"}
    rng = np.random.default_rng(0)
    # one window = one remote IP's 10 features
    X = pd.DataFrame(rng.uniform(0, 500, size=(1, len(live_classifier.FEATURES))),
                     columns=live_classifier.FEATURES)
    row = {f: float(X.iloc[0][f]) for f in live_classifier.FEATURES}

    classify_ms = time_it(lambda: live_classifier.classify(clf, row, 0), n=50)
    # deviation explanation is pure arithmetic on 10 numbers
    center = np.zeros(len(live_classifier.FEATURES)) + 1
    spread = np.ones(len(live_classifier.FEATURES))
    def deviation():
        z = (np.log1p(X.to_numpy()) - center) / spread
        return np.clip(z, 0, None)
    dev_ms = time_it(deviation, n=50)
    return {
        "classify_per_window_ms": round(classify_ms, 3),
        "deviation_explain_per_window_ms": round(dev_ms, 4),
        "note": "live explanation is exact arithmetic, far cheaper than SHAP",
    }


if __name__ == "__main__":
    print("Benchmarking replay (Random Forest + TreeSHAP) ...")
    replay = bench_replay()
    print("Benchmarking live (Decision Tree classifier + deviation) ...")
    live = bench_live()

    result = {"replay": replay, "live": live}
    (ARTIFACTS / "overhead.json").write_text(json.dumps(result, indent=2))

    print("\n=== Replay mode ===")
    print(f"  Flows scored:            {replay['flows_scored']:,}")
    print(f"  Detection (model only):  {replay['detect_per_flow_ms']:.4f} ms / flow")
    print(f"  Alerts explained (SHAP): {replay['alerts_explained']:,}")
    print(f"  SHAP explanation:        {replay['shap_per_alert_ms']:.2f} ms / alert")
    print(f"  Overhead per alert:      {replay['overhead_factor_per_alert']}x detection")
    print("\n=== Live mode ===")
    print(f"  Classify attack type:    {live['classify_per_window_ms']:.3f} ms / window")
    print(f"  Deviation explanation:   {live['deviation_explain_per_window_ms']:.4f} ms / window")
    print(f"\nSaved to {ARTIFACTS / 'overhead.json'}")
