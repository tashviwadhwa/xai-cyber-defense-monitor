# XAI Cyber Defense Monitor

A multi-source threat monitoring system with explainable alerts. It combines network traffic with login activity, gives every event a 0–100 risk score, maps it to MITRE ATT&CK, recommends a response, and explains *why* each alert fired so an analyst can act on it with confidence.

## Contents

- [Two modes](#two-modes)
- [Features](#features)
- [Quick start](#quick-start)
- [Live capture](#live-capture-two-laptops)
- [How the risk score works](#how-the-risk-score-works)
- [Results](#results)
- [Limitations](#limitations)
- [Project structure](#project-structure)
- [Configuration](#configuration)
- [Demo walkthrough](#demo-walkthrough)
- [Dataset and citation](#dataset-and-citation)

## Two modes

|              | Replay demo                                                                     | Live capture                                                             |
|--------------|---------------------------------------------------------------------------------|--------------------------------------------------------------------------|
| Network data | UNSW-NB15: packets recorded on a lab testbed, attacks made by a simulation tool | **Real packets** captured on your laptop                                 |
| Login data   | Simulated                                                                       | **Real** failed logins from Windows (event 4625) or Linux (`auth.log`)   |
| Detection    | Random Forest trained on labelled attacks (supervised)                          | Behavioural baseline learned from your own normal traffic (unsupervised) |
| Explanation  | SHAP (TreeExplainer)                                                            | Exact per-feature deviation from normal                                  |
| Real time?   | No, a replay of pre-scored data                                                 | **Yes**                                                                  |

## Features

- **Detection:** supervised Random Forest (replay) and an unsupervised behavioural baseline (live).
- **Correlation:** network evidence, failed logins and repeat alerts from the same IP within 15 minutes are combined into one risk score.
- **Explainability:** SHAP for the Random Forest; exact deviation from normal for live mode.
- **Live attack-type classifier:** a trained Decision Tree names each live alert (Port scan / Traffic flood / Brute force / Normal) instead of using hand-written rules. It reaches 99% accuracy on held-out windows (`live_classifier.py`).
- **Source reputation lookup:** each alerting IP is enriched with its country, network owner and abuse-confidence score. Only public IPs are looked up; private and lab IPs are never sent out (`ip_reputation.py`).
- **MITRE ATT&CK mapping and a recommended response** for every alert.
- **Analyst actions:**
  - Block an IP (replay: later alerts from it are marked *Blocked*; live: shows the real Windows Firewall command).
  - Export alerts as CSV.
  - Download an incident report.
- **Model comparison:** 4 algorithms benchmarked on the same test set.

## Quick start

Requires Python 3.10–3.13.

```bash
git clone https://github.com/tashviwadhwa/xai-cyber-defense-monitor.git
cd xai-cyber-defense-monitor

python -m venv venv
venv\Scripts\activate            # Windows
source venv/bin/activate         # macOS / Linux

pip install -r requirements.txt
```

For live capture on Windows, also install **[Npcap](https://npcap.com/#download)** (keep the default options).

### Run the replay demo

```bash
python run_pipeline.py     # trains 4 models, simulates logs, scores and explains (~3 min)
streamlit run app.py
```

## Live capture (two laptops)

Both laptops must be on the same Wi-Fi network or hotspot.

> **Responsible use:** only run `attack_demo.py` against devices you own.

**Laptop A (monitored):** open a terminal **as Administrator** (on Linux, run with `sudo`) and start the monitor:

```bash
python live_monitor.py --learn 10
```

It prints this laptop's IP. For 10 minutes, use the laptop normally (browse, watch a video) while it learns what normal looks like. When it prints `DETECTING`, open a second terminal, run `streamlit run app.py`, and go to the **Live capture** tab.

**Laptop B (attacker):** a normal terminal is enough.

```bash
python attack_demo.py <laptop A's IP> scan     # port scan
python attack_demo.py <laptop A's IP> brute    # wrong-password logins over SMB (Windows)
python attack_demo.py <laptop A's IP> all      # both together, which should reach Critical
```

Later runs of `live_monitor.py` reuse the saved baseline. Add `--relearn` to learn again, for example on a different network.

### Troubleshooting

- **No traffic from laptop B at all:** some phone hotspots isolate devices from each other. Try another hotspot or a Wi-Fi router, and check that `ping <laptop A's IP>` works from laptop B.
- **No failed logins:** make sure Windows records them. In an Administrator terminal run `auditpol /set /subcategory:"Logon" /failure:enable`.
- **SMB brute force doesn't reach laptop A:** the network needs to be set to *Private* with file sharing on. If that isn't possible, type a wrong password at laptop A's lock screen 10 times instead. It appears as a brute force from "local console".
- **Wrong network interface:** run `python live_monitor.py --iface "Wi-Fi"` (use your adapter's name).

## How the risk score works

```
risk = 100 × (0.60 × network evidence + 0.25 × failed logins + 0.15 × repeat alerts)
```

| Component        | Replay                                                                            | Live                                          |
|------------------|-----------------------------------------------------------------------------------|-----------------------------------------------|
| Network evidence | The model's attack probability                                                    | How far traffic is above this laptop's normal |
| Failed logins    | From the same IP in the last 15 minutes; 10 or more counts as full               | Same                                          |
| Repeat alerts    | Earlier alerts from the same IP in the last 15 minutes; 5 or more counts as full | Same                                          |

**Severity bands:** Critical ≥ 80, High ≥ 60, Medium ≥ 40. Medium and above become alerts. Live mode adds one rule: 10 or more failed logins in 15 minutes is at least High.

## Results

**Replay model** (held-out test set, 52,603 flows)

| Model                      | Type accuracy | Detection rate | False-alarm rate |
|----------------------------|---------------|----------------|------------------|
| Logistic Regression        | 76.8%         | 96.7%          | 11.9%            |
| Decision Tree              | 81.4%         | 98.4%          | 14.0%            |
| Gradient Boosting          | 82.4%         | 96.7%          | 8.6%             |
| **Random Forest (chosen)** | **82.4%**     | **97.0%**      | **8.4%**         |

*Type accuracy* is how often the attack category is named correctly. *Detection rate* is the share of attacks flagged. *False-alarm rate* is the share of normal flows wrongly flagged.

With correlation applied in the replay, 99.6% of real attacks raised an alert and only 3.7% of normal flows did.

**Live mode** (tested with a real port scan and brute-force attempts): a 1,000-port scan scored about 9 "spreads" above normal (anything above 3 is unusual) and was labelled *Port scan*. Brute-force sources were flagged High, and normal traffic stayed at a score of 0.

## Limitations

- The replay's login log and IP addresses are simulated; UNSW-NB15 has neither.
- In the replay simulation, attacker IPs only send attack traffic, which makes the repeat-alert score stronger than it would be on a real network.
- DoS, Exploits, Analysis and Backdoor look very similar in UNSW-NB15, so the model often confuses their type while still flagging them as attacks.
- Features such as source TTL are unusually predictive in UNSW-NB15 because of how the testbed was built, so real-world accuracy would be lower.
- Live mode monitors one machine only and learns "normal" from a short window, so unusual but harmless activity (for example a large game download) can raise alerts.

**Design note:** an Isolation Forest was tried first for live mode. It scored a real port scan barely above normal browsing, because it can't tell "far beyond anything seen" from "at the edge of what was seen". It was replaced with the robust baseline, which is also exactly explainable.

## Project structure

| File                                        | What it does                                                  |
|---------------------------------------------|---------------------------------------------------------------|
| `common.py`                                 | Shared settings, dataset loading, MITRE ATT&CK mapping        |
| `step1_train_model.py`                      | Compares 4 algorithms, trains and evaluates the Random Forest |
| `step2_simulate_logs.py`                    | Builds the replay feed and the simulated login log            |
| `step3_score_and_explain.py`                | Correlation, risk scoring and SHAP explanations (replay)      |
| `live_classifier.py`                        | Trains and applies the live attack-type classifier            |
| `ip_reputation.py`                          | Source IP reputation and geolocation lookup                   |
| `make_result_figures.py`                    | ROC, precision-recall and model-comparison figures            |
| `benchmark_overhead.py`                     | Measures the monitor's performance overhead                   |
| `run_pipeline.py`                           | Runs the full replay build in order                           |
| `live_monitor.py`                           | Real-time capture, baseline learning, detection and scoring   |
| `login_reader.py`                           | Reads real failed logins from Windows or Linux                |
| `attack_demo.py`                            | Port scan, brute force and flood, for testing your own laptop |
| `app.py`                                    | Dashboard entry point: sidebar and tabs                       |
| `ui_replay.py`, `ui_live.py`, `ui_model.py` | One file per dashboard tab                                    |
| `ui_common.py`                              | Charts, response panel, incident report                       |
| `data/`, `artifacts/`, `.streamlit/`        | Datasets, trained models and outputs, Streamlit config        |

## Configuration

**AbuseIPDB reputation score (optional).** Get a free key from [abuseipdb.com](https://www.abuseipdb.com) and set it before running the monitor:

```bash
set ABUSEIPDB_KEY=your_key            # Windows (cmd)
$env:ABUSEIPDB_KEY="your_key"         # Windows (PowerShell)
export ABUSEIPDB_KEY=your_key         # macOS / Linux
```

Without a key, geolocation still works and the abuse score is simply skipped.

## Demo walkthrough

A 5-minute tour of the dashboard:

1. **Replay tab:** press Play, then pause and click a Critical alert. Look at the SHAP chart, the risk breakdown, the IP timeline and the MITRE mapping.
2. Find `203.0.113.41` in "Riskiest source IPs". Its timeline shows a password brute force, a successful login, and attacks continuing afterwards: a likely account takeover.
3. Click **Block** on an attacker IP. Its later alerts become *Blocked* and the open-alert count drops.
4. **An honest edge case:** `10.0.6.186` is an employee locked out of their account (9 failed logins). Most of their traffic stays Low, but at about 09:43 the model wrongly flags one flow as Fuzzers and the failed logins push it to Medium. The timeline shows an internal IP with no other alerts, which is how an analyst would clear it.
5. **Live tab:** run `attack_demo.py <laptop A's IP> all` from laptop B. Within seconds a Port scan alert appears, explained as "N refused connections (normal = 0), N ports touched (normal = 1)". It escalates when the brute force starts.
6. **Model report tab:** see why Random Forest was chosen, using the comparison table.

## Dataset and citation

Moustafa, N. and Slay, J. (2015). UNSW-NB15: a comprehensive data set for network intrusion detection systems. *Military Communications and Information Systems Conference (MilCIS)*, IEEE.