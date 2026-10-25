"""
simulate_incident.py  (Review 3 - case-study simulation)

Scaled-down replay of the 2024 Snowflake customer-credential data-theft
campaign (UNC5537, Mandiant) against the project's own detection code.

What is REAL here : the baseline training, the deviation scoring, the risk
                    formula, the brute-force rule and the attack-type
                    classifier -- all imported from live_monitor.py and
                    live_classifier.py, not re-implemented.
What is SIMULATED : the packets/logins. No traffic is captured; each stage
                    is injected as the 10-number window that live_monitor
                    would have computed from that traffic. The monitored
                    host stands in for the victim's data-access gateway.

Run:  python simulate_incident.py            -> prints tables, writes
      artifacts/incident_simulation.json
"""

import json

import numpy as np
import pandas as pd

import live_classifier
import live_monitor as lm
from common import ARTIFACTS, attack_info

SEEDS = range(30)
FEATURES = lm.FEATURES


# ---------------------------------------------------------------------------
# Baselines (what "normal" looked like before the attack)
# ---------------------------------------------------------------------------
def make_baseline(rng, streaming=True, n=240):
    rows = []
    for _ in range(n):
        if streaming and rng.random() < 0.15:          # video / large download
            pin = rng.integers(200, 800)
            rows.append(dict(pkts_in=pin, pkts_out=rng.integers(100, 400),
                             bytes_in=pin * rng.integers(900, 1400),
                             bytes_out=rng.integers(8_000, 30_000),
                             local_ports=1, remote_ports=1, syn_in=0,
                             rst_out=0, icmp_in=0))
        else:                                           # ordinary browsing
            pin = rng.integers(0, 60)
            pout = rng.integers(0, 60)
            rows.append(dict(pkts_in=pin, pkts_out=pout,
                             bytes_in=pin * rng.integers(60, 1200),
                             bytes_out=pout * rng.integers(60, 900),
                             local_ports=rng.integers(1, 3),
                             remote_ports=rng.integers(1, 4),
                             syn_in=rng.integers(0, 3),
                             rst_out=rng.integers(0, 2), icmp_in=0))
    df = pd.DataFrame(rows)
    tot = df.pkts_in + df.pkts_out
    df["avg_pkt_size"] = ((df.bytes_in + df.bytes_out) / tot.where(tot > 0, 1)).fillna(0)
    df.insert(0, "src_ip", "x")
    return lm.train_baseline(df)


# ---------------------------------------------------------------------------
# Attack-stage windows (10 live features each)
# ---------------------------------------------------------------------------
def window(rng, pin, pout, bin_, bout, lports=1, rports=1, syn=1, rst=0):
    tot = max(pin + pout, 1)
    return dict(pkts_in=pin, pkts_out=pout, bytes_in=bin_, bytes_out=bout,
                local_ports=lports, remote_ports=rports, syn_in=syn,
                rst_out=rst, icmp_in=0, avg_pkt_size=(bin_ + bout) / tot)


def small_session(rng):
    """One short HTTPS/SQL session: a login or a few metadata queries."""
    pin = int(rng.integers(25, 55))
    pout = int(rng.integers(25, 55))
    return window(rng, pin, pout, pin * 420, pout * 380)


def exfil_window(rng, scale=1.0):
    """GET of staged files: bulk data flows OUT of the host to the attacker."""
    pout = int(3000 * scale * rng.uniform(0.9, 1.1))
    pin = int(pout * 0.5)                       # TCP ACKs coming back
    return window(rng, pin, pout, pin * 66, pout * 1400, syn=0)


# ---------------------------------------------------------------------------
# Scoring: identical formula to live_monitor.main()
# ---------------------------------------------------------------------------
def score(base, clf, feat, failed, prior):
    df = pd.DataFrame([feat])
    z = lm.deviations(df, base)
    max_dev = float(z.max(axis=1).iloc[0])
    net = float(np.clip((max_dev - lm.Z_START) / (lm.Z_FULL - lm.Z_START), 0, 1))
    if feat["pkts_in"] + feat["pkts_out"] == 0:
        net = 0.0
    risk = round(100 * (0.60 * net + 0.25 * min(failed / 10, 1) + 0.15 * min(prior / 5, 1)))
    if failed >= lm.BRUTE_FORCE_RULE:
        risk = max(risk, 60)
    sev = ("Critical" if risk >= 80 else "High" if risk >= 60
           else "Medium" if risk >= 40 else "Low")
    alert = risk >= 40
    label = ""
    top = ""
    if alert:
        row = pd.Series(feat)
        pred = live_classifier.classify(clf, row, failed)
        label = pred if pred and pred != "Unusual traffic" else \
            lm.label_window(pd.Series(feat), failed, base)
        top = z.iloc[0].idxmax() if net > 0 else "failed logins"
    return dict(max_deviation=round(max_dev, 2), risk=int(risk), severity=sev,
                alert=alert, label=label, top_feature=top)


def run_stage_windows(base, clf, windows, failed_seq, state):
    out = []
    for w, f in zip(windows, failed_seq):
        r = score(base, clf, w, f, state["prior"])
        if r["alert"]:
            state["prior"] += 1
        out.append(r)
    return out


def stage_table(seed, streaming=True):
    rng = np.random.default_rng(seed)
    base = make_baseline(rng, streaming)
    clf = live_classifier.load()
    st = {"prior": 0}
    rows = []

    def add(stage, desc, windows, failed_seq):
        for i, r in enumerate(run_stage_windows(base, clf, windows, failed_seq, st)):
            rows.append(dict(stage=stage, window=i + 1, desc=desc, **r))

    # S1a: valid stolen credentials, first-try success (the common case in the
    # real campaign). Successful logins are not read by login_reader.py, so
    # failed-login count is 0.
    add("S1a", "Valid-credential login from VPN IP", [small_session(rng)], [0])
    # S1b / S1c: sensitivity variants -- stale credentials tried first.
    add("S1b", "Stale credentials: 6 failed logins then success",
        [small_session(rng)], [6])
    add("S1c", "Stale credentials: 12 failed logins then success",
        [small_session(rng)], [12])
    # S2: reconnaissance SQL (SHOW TABLES, LIST) -- small, normal-looking.
    add("S2", "Recon queries (SHOW TABLES / LIST)",
        [small_session(rng) for _ in range(3)], [0, 0, 0])
    # S4: bulk exfiltration, 6 consecutive 5-second windows at full scale.
    add("S4", "Bulk exfiltration (GET) at full scale",
        [exfil_window(rng, 1.0) for _ in range(6)], [0] * 6)
    return rows


def exfil_sweep(seed, streaming=True):
    """Lowest exfiltration rate that still raises an alert."""
    rng = np.random.default_rng(seed)
    base = make_baseline(rng, streaming)
    clf = live_classifier.load()
    out = {}
    for scale in (1.0, 0.5, 0.25, 0.10, 0.05, 0.02, 0.01):
        st = {"prior": 0}
        res = run_stage_windows(base, clf, [exfil_window(rng, scale) for _ in range(6)],
                                [0] * 6, st)
        out[scale] = dict(first_window_alert=res[0]["alert"],
                          any_alert_in_6=any(r["alert"] for r in res),
                          peak_risk=max(r["risk"] for r in res))
    return out


def main():
    summary = {}
    for name, streaming in (("baseline_browsing_plus_streaming", True),
                            ("baseline_browsing_only", False)):
        # Detailed single run (seed 42) for the per-window table
        detail = stage_table(42, streaming)
        # Stability across seeds
        per_stage = {}
        for s in SEEDS:
            for r in stage_table(s, streaming):
                key = r["stage"]
                d = per_stage.setdefault(key, dict(n=0, alerts=0, max_risk=[], labels=[]))
                d["n"] += 1
                d["alerts"] += int(r["alert"])
                d["max_risk"].append(r["risk"])
                if r["alert"]:
                    d["labels"].append(r["label"])
        stability = {k: dict(alert_rate=round(v["alerts"] / v["n"], 3),
                             median_risk=int(np.median(v["max_risk"])),
                             labels=sorted(set(v["labels"]))) for k, v in per_stage.items()}
        sweeps = [exfil_sweep(s, streaming) for s in SEEDS]
        sweep_sum = {}
        for scale in sweeps[0]:
            sweep_sum[str(scale)] = dict(
                first_window_alert_rate=round(np.mean([w[scale]["first_window_alert"] for w in sweeps]), 3),
                any_alert_rate=round(np.mean([w[scale]["any_alert_in_6"] for w in sweeps]), 3),
                median_peak_risk=int(np.median([w[scale]["peak_risk"] for w in sweeps])))
        summary[name] = dict(detail_seed42=detail, stability_30_seeds=stability,
                             exfil_rate_sweep=sweep_sum)

    (ARTIFACTS / "incident_simulation.json").write_text(json.dumps(summary, indent=2, default=str))

    for name, s in summary.items():
        print("=" * 78)
        print(name)
        print("-- seed 42, per window --")
        print(pd.DataFrame(s["detail_seed42"])[["stage", "window", "max_deviation", "risk",
                                                "severity", "alert", "label", "top_feature"]].to_string(index=False))
        print("-- 30 seeds: alert rate per stage --")
        print(pd.DataFrame(s["stability_30_seeds"]).T.to_string())
        print("-- exfil rate sweep (fraction of full rate) --")
        print(pd.DataFrame(s["exfil_rate_sweep"]).T.to_string())
    print("\nMITRE mapping of the labels used:", attack_info("Traffic flood"))


if __name__ == "__main__":
    main()
