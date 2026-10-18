"""
STEP 3 - Score, correlate and explain.

For every flow in the replay feed:
  1. The model gives an attack probability and an attack type.
  2. We look up the same source IP in the login log (last 15 minutes)
     and in the model's own earlier alerts (last 15 minutes).
  3. We combine the three signals into one 0-100 risk score.
  4. For every alert, SHAP tells us which traffic features pushed
     the model towards "attack".

Risk score formula (explain this one in the viva):

    risk = 100 x ( 0.60 x network_score      model's attack probability
                 + 0.25 x login_score        failed logins from this IP, capped at 10
                 + 0.15 x repeat_score )     earlier alerts from this IP, capped at 5

    Critical >= 80   High >= 60   Medium >= 40   Low < 40
    Anything Medium or above becomes an alert.

So a single confident detection lands in High (about 60), but the same
detection from an IP that is also brute-forcing passwords and has
already triggered alerts climbs to Critical. A user who only mistyped
their password many times stays Low, because the network model sees
nothing wrong.

Run:  python step3_score_and_explain.py   (about 1 minute)
"""

import json
import time

import joblib
import numpy as np
import pandas as pd
import shap

from common import ARTIFACTS, TARGET, nice_name, to_features

WINDOW = pd.Timedelta(minutes=15)
W_NETWORK, W_LOGIN, W_REPEAT = 0.60, 0.25, 0.15
LOGIN_CAP, REPEAT_CAP = 10, 5
ALERT_THRESHOLD = 40

# 1. Load everything ---------------------------------------------------------
model = joblib.load(ARTIFACTS / "model.joblib")
encoder = joblib.load(ARTIFACTS / "encoder.joblib")
with open(ARTIFACTS / "metrics.json") as f:
    feature_cols = json.load(f)["feature_cols"]

feed = pd.read_csv(ARTIFACTS / "feed_flows.csv", parse_dates=["timestamp"])
auth = pd.read_csv(ARTIFACTS / "auth_log.csv", parse_dates=["timestamp"])
X = to_features(feed, encoder)[feature_cols]
classes = list(model.classes_)
normal_idx = classes.index("Normal")

# 2. Network signal: what does the model think? -----------------------------
proba = model.predict_proba(X)                      # one column per class
feed["pred_cat"] = [classes[i] for i in proba.argmax(axis=1)]
feed["confidence"] = proba.max(axis=1)
feed["attack_prob"] = 1 - proba[:, normal_idx]      # P(anything but Normal)

# The attack type we will explain = most likely NON-normal class
attack_proba = proba.copy()
attack_proba[:, normal_idx] = -1
explain_idx = attack_proba.argmax(axis=1)
feed["explained_class"] = [classes[i] for i in explain_idx]


# 3. Correlation: look back 15 minutes for the same IP -----------------------
def count_in_window(event_times_by_ip, ips, times, include_now):
    """For each (ip, time) count events from that ip in the 15 min before time."""
    counts = np.zeros(len(ips), dtype=int)
    for i, (ip, t) in enumerate(zip(ips, times)):
        ev = event_times_by_ip.get(ip)
        if ev is None:
            continue
        side = "right" if include_now else "left"
        hi = np.searchsorted(ev, np.datetime64(t), side=side)
        lo = np.searchsorted(ev, np.datetime64(t - WINDOW), side="left")
        counts[i] = hi - lo
    return counts


failed = auth[auth.outcome == "FAILED"]
failed_by_ip = {ip: np.sort(g.timestamp.values) for ip, g in failed.groupby("src_ip")}
feed["failed_logins_15m"] = count_in_window(failed_by_ip, feed.src_ip, feed.timestamp, True)

flagged = feed[feed.pred_cat != "Normal"]
flagged_by_ip = {ip: np.sort(g.timestamp.values) for ip, g in flagged.groupby("src_ip")}
feed["prior_alerts_15m"] = count_in_window(flagged_by_ip, feed.src_ip, feed.timestamp, False)

# 4. Risk score ---------------------------------------------------------------
feed["network_score"] = feed.attack_prob
feed["login_score"] = np.minimum(feed.failed_logins_15m / LOGIN_CAP, 1)
feed["repeat_score"] = np.minimum(feed.prior_alerts_15m / REPEAT_CAP, 1)
feed["risk"] = (100 * (W_NETWORK * feed.network_score
                       + W_LOGIN * feed.login_score
                       + W_REPEAT * feed.repeat_score)).round().astype(int)
feed["severity"] = pd.cut(feed.risk, bins=[-1, 39, 59, 79, 100],
                          labels=["Low", "Medium", "High", "Critical"]).astype(str)
feed["is_alert"] = feed.risk >= ALERT_THRESHOLD

# 5. SHAP explanations for alerts ---------------------------------------------
alert_rows = np.where(feed.is_alert)[0]
print(f"Explaining {len(alert_rows):,} alerts with SHAP (about a minute) ...")
start = time.time()
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X.iloc[alert_rows])   # shape: rows x features x classes
print(f"  done in {time.time() - start:.0f}s")

reasons_col = [""] * len(feed)
top_reason_col = [""] * len(feed)
global_rows = []

for k, row in enumerate(alert_rows):
    cls = explain_idx[row]
    contrib = shap_values[k, :, cls]                 # this row, all features, its class
    order = np.argsort(-np.abs(contrib))[:6]         # 6 biggest effects either way
    reasons = [{
        "feature": feature_cols[j],
        "label": nice_name(feature_cols[j]),
        "value": float(feed.iloc[row][feature_cols[j]]) if feature_cols[j] not in ("proto", "service", "state")
        else str(feed.iloc[row][feature_cols[j]]),
        "contribution": float(contrib[j]),
    } for j in order]
    reasons_col[row] = json.dumps(reasons)
    top = max(reasons, key=lambda r: r["contribution"])
    top_reason_col[row] = top["label"]
    global_rows.append(np.abs(contrib))

feed["reasons"] = reasons_col
feed["top_reason"] = top_reason_col

# Global view: which features matter most across all alerts
global_imp = (pd.DataFrame({"feature": feature_cols,
                            "mean_abs_shap": np.mean(global_rows, axis=0)})
              .sort_values("mean_abs_shap", ascending=False))
global_imp["label"] = global_imp.feature.map(nice_name)

# 6. Save ---------------------------------------------------------------------
keep = ["flow_id", "timestamp", "src_ip", "proto", "service", "state", "dur",
        "sbytes", "dbytes", TARGET, "pred_cat", "confidence", "attack_prob",
        "explained_class", "failed_logins_15m", "prior_alerts_15m",
        "network_score", "login_score", "repeat_score", "risk", "severity",
        "is_alert", "top_reason", "reasons"]
feed[keep].rename(columns={TARGET: "true_cat"}).to_csv(ARTIFACTS / "scored_feed.csv", index=False)
global_imp.to_csv(ARTIFACTS / "global_importance.csv", index=False)

print(feed.severity.value_counts().reindex(["Critical", "High", "Medium", "Low"]).to_string())
truly_attack = feed[TARGET] != "Normal"
print(f"\nAlerts: {feed.is_alert.sum():,}  |  real attacks alerted: "
      f"{(feed.is_alert & truly_attack).sum() / truly_attack.sum():.1%}  |  "
      f"normal flows alerted: {(feed.is_alert & ~truly_attack).sum() / (~truly_attack).sum():.1%}")
print("Saved scored_feed.csv and global_importance.csv")
