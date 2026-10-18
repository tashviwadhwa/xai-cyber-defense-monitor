"""
STEP 2 - Build the second data source (login logs).

UNSW-NB15 only has network traffic. A real security team would also see
authentication logs (who tried to log in, did it fail). We simulate one
hour of activity:

  * Pick FEED_SIZE flows from the test set and give each a timestamp.
  * Give each flow a source IP. Normal traffic comes from ~150 internal
    hosts (10.x.x.x). Attack traffic comes from 14 external attacker IPs
    (203.0.113.x / 198.51.100.x are reserved "documentation" ranges, so
    they can never belong to a real person).
  * Generate login events for those IPs: some attackers run password
    brute-force bursts, normal users occasionally mistype a password,
    and one employee is locked out (a deliberate "noisy but innocent"
    case to talk about in the demo).

The simulation is SIMULATED - say so in the presentation.

Run:  python step2_simulate_logs.py
"""

import numpy as np
import pandas as pd

from common import ARTIFACTS, FEED_SIZE, RANDOM_STATE, TARGET

rng = np.random.default_rng(RANDOM_STATE)
START = pd.Timestamp("2026-09-18 09:00:00")
HOUR_SECONDS = 3600

# 1. Pick flows and give them timestamps ------------------------------------
test_df = pd.read_csv(ARTIFACTS / "test_split.csv")
feed = test_df.sample(n=FEED_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)

# sample() already returns rows in random order, so attack types end up
# spread across the hour. We just hand out increasing timestamps.
offsets = np.sort(rng.uniform(0, HOUR_SECONDS, size=FEED_SIZE))
feed["timestamp"] = (START + pd.to_timedelta(offsets, unit="s")).floor("s")

# 2. Assign source IPs ------------------------------------------------------
internal_ips = [f"10.0.{rng.integers(1, 20)}.{rng.integers(2, 250)}" for _ in range(150)]
internal_ips = list(dict.fromkeys(internal_ips))          # remove duplicates

attacker_ips = [f"203.0.113.{i}" for i in (7, 23, 41, 58, 77, 90, 112)] + \
               [f"198.51.100.{i}" for i in (5, 19, 33, 64, 88, 120, 201)]

attack_types = [c for c in feed[TARGET].unique() if c != "Normal"]
# Each attack type is carried out by 2-3 attackers, so one attacker IP
# shows up across several flows (that is what makes correlation useful).
crew = {a: list(rng.choice(attacker_ips, size=rng.integers(2, 4), replace=False))
        for a in attack_types}


def pick_ip(attack_cat):
    if attack_cat == "Normal":
        return rng.choice(internal_ips)
    return rng.choice(crew[attack_cat])


feed["src_ip"] = [pick_ip(a) for a in feed[TARGET]]
feed["flow_id"] = np.arange(1, len(feed) + 1)

# 3. Generate login events --------------------------------------------------
events = []


def add(ip, when, user, outcome):
    events.append({"timestamp": when, "src_ip": ip, "user": user, "outcome": outcome})


def rand_time(lo=0, hi=HOUR_SECONDS):
    return START + pd.Timedelta(seconds=int(rng.uniform(lo, hi)))


# Normal staff: a few successful logins, sometimes one or two typos first
for i, ip in enumerate(internal_ips):
    user = f"emp{i:03d}"
    for _ in range(rng.integers(1, 4)):
        t = rand_time()
        if rng.random() < 0.12:
            for k in range(rng.integers(1, 3)):
                add(ip, t - pd.Timedelta(seconds=20 * (k + 1)), user, "FAILED")
        add(ip, t, user, "SUCCESS")

# One locked-out employee: many failures but completely normal traffic
lockout_ip = internal_ips[3]
t0 = rand_time(1200, 2400)
for k in range(9):
    add(lockout_ip, t0 + pd.Timedelta(seconds=35 * k), "emp003", "FAILED")

# Attackers: about 60% of them brute-force passwords during the hour
targets = ["admin", "root", "backup", "svc_web", "oracle", "test"]
for ip in attacker_ips:
    if rng.random() < 0.6:
        n_bursts = rng.integers(1, 4)
        for _ in range(n_bursts):
            t = rand_time(0, HOUR_SECONDS - 300)
            for k in range(rng.integers(8, 40)):
                add(ip, t + pd.Timedelta(seconds=int(rng.uniform(0, 240))),
                    rng.choice(targets), "FAILED")
        # a couple of them eventually guess right = likely account takeover
        if rng.random() < 0.3:
            add(ip, t + pd.Timedelta(seconds=250), rng.choice(targets), "SUCCESS")

auth = pd.DataFrame(events).sort_values("timestamp").reset_index(drop=True)

# 4. Save -----------------------------------------------------------------
feed.to_csv(ARTIFACTS / "feed_flows.csv", index=False)
auth.to_csv(ARTIFACTS / "auth_log.csv", index=False)

print(f"Feed: {len(feed):,} flows from {feed.src_ip.nunique()} source IPs")
print(f"Login log: {len(auth):,} events "
      f"({(auth.outcome == 'FAILED').sum():,} failed, {(auth.outcome == 'SUCCESS').sum():,} successful)")
print("Saved feed_flows.csv and auth_log.csv")
