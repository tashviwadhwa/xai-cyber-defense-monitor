"""
LIVE MODE - real-time monitoring of THIS computer.

What it does, every few seconds:
  1. Captures real network packets (scapy) and groups them by the other
     computer involved (the "remote IP").
  2. Turns each remote IP's packets into 10 numbers: packets in/out, bytes
     in/out, how many of our ports it touched, SYN packets, and so on.
  3. Reads real failed logins from the operating system (login_reader.py).
  4. Scores each remote IP against a behavioural baseline that learned
     what NORMAL looks like on this machine: for every feature, how far
     above the typical value is it, measured in "normal spreads"? The
     most abnormal feature sets the network score. That score is combined
     with failed logins and repeat alerts into a 0-100 risk.

Why not Isolation Forest (or SHAP) here? We tried it: an Isolation Forest
cannot tell "10x beyond anything seen in training" from "at the edge of
training", so a real port scan only scored about as odd as normal
browsing. The baseline below has no such ceiling, and its explanation is
exact: each bar IS that feature's deviation, nothing is approximated.

Two phases:
  LEARN  : first run records your normal activity (browse, stream, chat)
           for --learn minutes and trains the baseline.
  DETECT : afterwards every window is scored against that baseline.

Run in an Administrator terminal (Windows) or with sudo (Linux/macOS):
    python live_monitor.py --learn 10        first time: learn, then detect
    python live_monitor.py                   later: detect with saved baseline
    python live_monitor.py --relearn         throw away the baseline, learn again
Then, in another terminal:  streamlit run app.py  ->  "Live monitor" tab.
Stop with Ctrl+C.
"""

import argparse
import json
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd

from common import ARTIFACTS
from login_reader import LoginReader
import live_classifier
import ip_reputation

LIVE_DIR = ARTIFACTS / "live"
LIVE_DIR.mkdir(exist_ok=True)
BASELINE_FILE = LIVE_DIR / "baseline.joblib"
WINDOWS_FILE = LIVE_DIR / "windows.csv"      # every scored window, appended
LOGINS_FILE = LIVE_DIR / "logins.csv"        # every failed login seen
STATUS_FILE = LIVE_DIR / "status.json"       # what the monitor is doing now

_FEATURES_DOC = "the ten live traffic features; live_classifier mirrors this list"
FEATURES = ["pkts_in", "pkts_out", "bytes_in", "bytes_out", "local_ports",
            "remote_ports", "syn_in", "rst_out", "icmp_in", "avg_pkt_size"]
# Keep the classifier's feature list identical to this one.
assert live_classifier.FEATURES == FEATURES, "live_classifier FEATURES out of sync"
FEATURE_LABELS = {
    "pkts_in": "Packets received from it",
    "pkts_out": "Packets sent to it",
    "bytes_in": "Bytes received from it",
    "bytes_out": "Bytes sent to it",
    "local_ports": "Different ports of ours it contacted",
    "remote_ports": "Different ports it used",
    "syn_in": "Connection attempts (SYN) from it",
    "rst_out": "Connections we refused (RST)",
    "icmp_in": "ICMP (ping) packets from it",
    "avg_pkt_size": "Average packet size",
}

WINDOW_15 = timedelta(minutes=15)
BRUTE_FORCE_RULE = 10        # failed logins in 15 min that force at least High


# ---------------------------------------------------------------------------
# 1. Packet capture (runs in a background thread)
# ---------------------------------------------------------------------------
class Capture:
    def __init__(self, iface):
        self.iface = iface
        self.packets = []
        self.total = 0
        self.error = None                  # set if the capture thread crashes
        self.lock = threading.Lock()

    def _on_packet(self, pkt):
        from scapy.all import ICMP, IP, TCP, UDP
        if IP not in pkt:
            return
        ip = pkt[IP]
        sport = dport = None
        flags = ""
        if TCP in pkt:
            proto, sport, dport, flags = "tcp", pkt[TCP].sport, pkt[TCP].dport, str(pkt[TCP].flags)
        elif UDP in pkt:
            proto, sport, dport = "udp", pkt[UDP].sport, pkt[UDP].dport
        elif ICMP in pkt:
            proto = "icmp"
        else:
            proto = "other"
        with self.lock:
            self.packets.append((ip.src, ip.dst, proto, sport, dport, flags, len(pkt)))
            self.total += 1

    def _run(self):
        from scapy.all import sniff
        try:
            sniff(iface=self.iface, prn=self._on_packet, store=False)
        except Exception as exc:           # remember it so the main loop can report it
            self.error = exc

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def take(self):
        """Hand over the packets captured since the last call."""
        if self.error:
            raise RuntimeError(f"packet capture stopped: {self.error}")
        with self.lock:
            batch, self.packets = self.packets, []
        return batch


# ---------------------------------------------------------------------------
# 2. Packets -> one row of features per remote IP
# ---------------------------------------------------------------------------
def window_features(packets, local_ips):
    stats = defaultdict(lambda: {"pkts_in": 0, "pkts_out": 0, "bytes_in": 0, "bytes_out": 0,
                                 "local_ports": set(), "remote_ports": set(), "syn_in": 0,
                                 "rst_out": 0, "icmp_in": 0})
    for src, dst, proto, sport, dport, flags, size in packets:
        if dst in local_ips and src not in local_ips:          # coming in
            s = stats[src]
            s["pkts_in"] += 1
            s["bytes_in"] += size
            if dport is not None:
                s["local_ports"].add(dport)
                s["remote_ports"].add(sport)
            if proto == "tcp" and "S" in flags and "A" not in flags:
                s["syn_in"] += 1
            if proto == "icmp":
                s["icmp_in"] += 1
        elif src in local_ips and dst not in local_ips:        # going out
            s = stats[dst]
            s["pkts_out"] += 1
            s["bytes_out"] += size
            if proto == "tcp" and "R" in flags:
                s["rst_out"] += 1

    rows = []
    for remote, s in stats.items():
        total_pkts = s["pkts_in"] + s["pkts_out"]
        rows.append({
            "src_ip": remote,
            "pkts_in": s["pkts_in"], "pkts_out": s["pkts_out"],
            "bytes_in": s["bytes_in"], "bytes_out": s["bytes_out"],
            "local_ports": len(s["local_ports"]), "remote_ports": len(s["remote_ports"]),
            "syn_in": s["syn_in"], "rst_out": s["rst_out"], "icmp_in": s["icmp_in"],
            "avg_pkt_size": (s["bytes_in"] + s["bytes_out"]) / total_pkts if total_pkts else 0,
        })
    return pd.DataFrame(rows, columns=["src_ip"] + FEATURES)


def to_model_input(df):
    # log1p squashes big counts (a 5 MB download vs a 5 KB page) onto a similar scale
    return np.log1p(df[FEATURES].astype(float))


# ---------------------------------------------------------------------------
# 3. Baseline: learn what normal looks like
# ---------------------------------------------------------------------------
MIN_SPREAD = 0.5     # on the log scale; stops "never changed" features exploding
Z_START, Z_FULL = 3, 8   # deviation where network score starts / reaches 1


def train_baseline(rows):
    """Learn the typical value and normal spread of every feature."""
    X = to_model_input(rows)
    q1, q3 = X.quantile(0.25), X.quantile(0.75)
    # IQR / 1.349 is a "robust standard deviation": one burst of odd
    # traffic during learning can't stretch it the way it would a normal std
    spread = ((q3 - q1) / 1.349).clip(lower=MIN_SPREAD)
    return {
        "center": X.median().to_dict(),
        "spread": spread.to_dict(),
        "typical": rows[FEATURES].median().to_dict(),       # shown in explanations
        "high": rows[FEATURES].quantile(0.99).to_dict(),    # used by the labelling rules
        "rows": len(rows),
        "learned_at": datetime.now().isoformat(timespec="seconds"),
    }


def deviations(df, base):
    """How many normal spreads ABOVE typical each feature is (0 if at or below)."""
    X = to_model_input(df)
    z = (X - pd.Series(base["center"])) / pd.Series(base["spread"])
    return z.clip(lower=0)[FEATURES]


def label_window(row, failed, base):
    """Name the kind of anomaly with simple, readable rules."""
    if row.local_ports >= max(15, 3 * base["high"]["local_ports"]):
        return "Port scan"
    if row.pkts_in >= max(300, 5 * base["high"]["pkts_in"]):
        return "Traffic flood"
    if failed >= 5:
        return "Brute force"
    return "Unusual traffic"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def write_status(**info):
    info["updated"] = datetime.now().isoformat(timespec="seconds")
    STATUS_FILE.write_text(json.dumps(info, indent=2))


def append_csv(df, path):
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def local_addresses(iface, override):
    if override:
        return {override}
    from scapy.all import get_if_addr
    return {get_if_addr(iface)}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Real-time monitor for this computer")
    ap.add_argument("--learn", type=float, default=10, help="minutes of normal activity to learn")
    ap.add_argument("--relearn", action="store_true", help="discard the saved baseline")
    ap.add_argument("--window", type=int, default=5, help="seconds per scoring window")
    ap.add_argument("--iface", default=None, help="network interface (default: scapy's choice)")
    ap.add_argument("--local-ip", default=None, help="this machine's IP, if auto-detect is wrong")
    ap.add_argument("--auth-log", default=None, help="Linux only: path to the auth log")
    args = ap.parse_args()

    from scapy.all import conf
    iface = args.iface or conf.iface
    local_ips = local_addresses(iface, args.local_ip)
    print(f"Interface: {iface}   This machine: {', '.join(local_ips)}")

    # Fail early and clearly if we are not allowed to capture
    from scapy.all import sniff
    sniff(iface=iface, timeout=1, store=False)

    capture = Capture(iface)
    capture.start()
    logins = LoginReader(linux_log=args.auth_log)
    logins.poll()                                   # sets status, skips old history

    # ---- LEARN phase -------------------------------------------------------
    if args.relearn or not BASELINE_FILE.exists():
        for f in (WINDOWS_FILE, LOGINS_FILE):
            f.unlink(missing_ok=True)
        end = datetime.now() + timedelta(minutes=args.learn)
        rows = []
        print(f"LEARNING normal behaviour for {args.learn:g} min. Use the computer normally "
              "(browse, watch a video, chat). Do NOT run attacks yet.")
        while datetime.now() < end:
            time.sleep(args.window)
            rows.append(window_features(capture.take(), local_ips))
            n = sum(len(r) for r in rows)
            left = (end - datetime.now()).total_seconds()
            write_status(phase="learning", iface=str(iface), local_ips=sorted(local_ips),
                         learn_until=end.isoformat(timespec="seconds"),
                         seconds_left=max(0, int(left)), baseline_rows=n,
                         packets=capture.total, login_status=logins.status)
            print(f"  {n} traffic samples, {capture.total:,} packets, {max(0, left):.0f}s left", end="\r")
        rows = pd.concat(rows, ignore_index=True)
        if len(rows) < 20:
            print("\nToo little traffic to learn from. Browse a few websites and run again "
                  "with a longer --learn time.")
            return
        baseline = train_baseline(rows)
        joblib.dump(baseline, BASELINE_FILE)
        print(f"\nBaseline learned from {len(rows)} samples.")
    else:
        baseline = joblib.load(BASELINE_FILE)
        print(f"Using saved baseline ({baseline['rows']} samples, learned {baseline['learned_at']}).")


    # ---- DETECT phase ------------------------------------------------------
    login_history = []                 # (time, ip) of every failed login
    alert_history = []                 # (time, ip) of every alert raised
    windows_done = 0

    # Feature 1: trained attack-type classifier (falls back to rules if absent)
    clf = live_classifier.load()
    if clf is not None:
        print("Attack-type classifier loaded (names each alert).")
    else:
        print("No attack-type classifier found; using rule-based labels. "
              "Run 'python live_classifier.py' to enable it.")

    print("DETECTING. Open the dashboard's Live monitor tab. Ctrl+C to stop.")
    while True:
        time.sleep(args.window)
        now = datetime.now()

        new_logins = logins.poll()
        if new_logins:
            append_csv(pd.DataFrame(new_logins), LOGINS_FILE)
            login_history += [(e["timestamp"], e["src_ip"]) for e in new_logins]

        feats = window_features(capture.take(), local_ips)
        # A source with new failed logins but no traffic this window still gets a row
        for ip in {e["src_ip"] for e in new_logins} - set(feats.src_ip):
            feats.loc[len(feats)] = [ip] + [0] * len(FEATURES)

        if len(feats):
            z = deviations(feats, baseline)
            feats["max_deviation"] = z.max(axis=1).round(2)
            # below 3 spreads = normal (0); 8 or more spreads = certainly abnormal (1)
            feats["network_score"] = ((feats.max_deviation - Z_START) / (Z_FULL - Z_START)).clip(0, 1)
            # login-only rows (no traffic at all) have no network evidence
            no_traffic = feats[["pkts_in", "pkts_out"]].sum(axis=1) == 0
            feats.loc[no_traffic, "network_score"] = 0.0

            cutoff = now - WINDOW_15
            login_history = [(t, ip) for t, ip in login_history if t >= cutoff]
            alert_history = [(t, ip) for t, ip in alert_history if t >= cutoff]
            feats["failed_logins_15m"] = [sum(ip == x for _, x in login_history) for ip in feats.src_ip]
            feats["prior_alerts_15m"] = [sum(ip == x for _, x in alert_history) for ip in feats.src_ip]

            feats["login_score"] = np.minimum(feats.failed_logins_15m / 10, 1)
            feats["repeat_score"] = np.minimum(feats.prior_alerts_15m / 5, 1)
            feats["risk"] = (100 * (0.60 * feats.network_score + 0.25 * feats.login_score
                                    + 0.15 * feats.repeat_score)).round().astype(int)
            # Rule, like a SIEM correlation rule: sustained password guessing is at least High
            brute = feats.failed_logins_15m >= BRUTE_FORCE_RULE
            feats.loc[brute, "risk"] = feats.loc[brute, "risk"].clip(lower=60)
            feats["severity"] = pd.cut(feats.risk, bins=[-1, 39, 59, 79, 100],
                                       labels=["Low", "Medium", "High", "Critical"]).astype(str)
            feats["is_alert"] = feats.risk >= 40

            # Feature 1: name each alert with the trained classifier; if it is
            # unavailable or unsure, fall back to the original rule-based label.
            def name_attack(r):
                if not r.is_alert:
                    return ""
                predicted = live_classifier.classify(clf, r._asdict(), r.failed_logins_15m)
                if predicted and predicted != "Unusual traffic":
                    return predicted
                return label_window(r, r.failed_logins_15m, baseline)
            feats["label"] = [name_attack(r) for r in feats.itertuples()]

            # Explanation: the features furthest above normal, biggest first
            contrib = z.to_numpy()
            reasons, tops = [], []
            for i, row in enumerate(feats.itertuples()):
                if not row.is_alert or row.network_score == 0:
                    reasons.append("")
                    tops.append("Failed logins" if row.is_alert else "")
                    continue
                order = [j for j in np.argsort(-contrib[i])[:6] if contrib[i, j] > 0]
                items = [{"feature": FEATURES[j], "label": FEATURE_LABELS[FEATURES[j]],
                          "value": float(getattr(row, FEATURES[j])),
                          "typical": float(baseline["typical"][FEATURES[j]]),
                          "contribution": float(contrib[i, j])} for j in order]
                reasons.append(json.dumps(items))
                tops.append(items[0]["label"])
            feats["reasons"] = reasons
            feats["top_reason"] = tops

            # Feature 2: look up reputation/geolocation for each alerting IP.
            feats["reputation"] = [ip_reputation.lookup(r.src_ip)["summary"]
                                   if r.is_alert else "" for r in feats.itertuples()]

            feats.insert(0, "timestamp", now.replace(microsecond=0))

            alert_history += [(now, ip) for ip in feats[feats.is_alert].src_ip]
            append_csv(feats, WINDOWS_FILE)

            for r in feats[feats.is_alert].itertuples():
                rep = f"  [{r.reputation}]" if r.reputation else ""
                print(f"[{now:%H:%M:%S}] {r.severity.upper():8} risk {r.risk:3}  {r.src_ip:15}  "
                      f"{r.label}  ({r.top_reason}){rep}")

        windows_done += 1
        write_status(phase="detecting", iface=str(iface), local_ips=sorted(local_ips),
                     windows=windows_done, packets=capture.total,
                     baseline_rows=baseline["rows"], login_status=logins.status,
                     window_seconds=args.window)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        write_status(phase="stopped")
        print("\nStopped.")
    except (PermissionError, OSError, RuntimeError) as exc:
        print(f"Could not start packet capture: {exc}\n"
              "Run the terminal as Administrator (Windows, with Npcap installed) "
              "or use sudo (Linux/macOS).")
