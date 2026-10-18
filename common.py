"""
common.py
Settings and helper functions shared by every script in the project.
Keeping them in one place means a change here applies everywhere.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

# ---------------------------------------------------------------------------
# Folders and files
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent          # the folder this file lives in
DATA_DIR = ROOT / "data"              # raw dataset CSVs go here
ARTIFACTS = ROOT / "artifacts"        # everything the scripts produce goes here
ARTIFACTS.mkdir(exist_ok=True)

TRAIN_FILE = DATA_DIR / "UNSW_NB15_training-set.csv"
TEST_FILE = DATA_DIR / "UNSW_NB15_testing-set.csv"   # optional, see load_splits()

# ---------------------------------------------------------------------------
# Dataset columns
# ---------------------------------------------------------------------------
TARGET = "attack_cat"                          # what we predict (Normal, DoS, ...)
CATEGORICAL = ["proto", "service", "state"]    # text columns that need encoding
DROP = ["id", "label", "attack_cat"]           # not used as model inputs

RANDOM_STATE = 42     # fixed seed so every run gives the same results
FEED_SIZE = 3000      # how many test flows the dashboard replays

# Plain-English names for the features, used in the dashboard's
# "why was this flagged" panel. Source: UNSW_NB15_features.csv
FEATURE_LABELS = {
    "dur": "Connection duration",
    "proto": "Protocol",
    "service": "Service (http, dns, ...)",
    "state": "Connection state",
    "spkts": "Packets sent by source",
    "dpkts": "Packets sent by destination",
    "sbytes": "Bytes sent by source",
    "dbytes": "Bytes sent by destination",
    "rate": "Packet rate",
    "sttl": "Source TTL (time-to-live)",
    "dttl": "Destination TTL",
    "sload": "Source bits per second",
    "dload": "Destination bits per second",
    "sloss": "Source packets lost",
    "dloss": "Destination packets lost",
    "sinpkt": "Gap between source packets",
    "dinpkt": "Gap between destination packets",
    "sjit": "Source jitter",
    "djit": "Destination jitter",
    "swin": "Source TCP window",
    "stcpb": "Source TCP sequence number",
    "dtcpb": "Destination TCP sequence number",
    "dwin": "Destination TCP window",
    "tcprtt": "TCP round-trip time",
    "synack": "SYN to SYN-ACK time",
    "ackdat": "SYN-ACK to ACK time",
    "smean": "Avg packet size from source",
    "dmean": "Avg packet size from destination",
    "trans_depth": "HTTP pipeline depth",
    "response_body_len": "HTTP response size",
    "ct_srv_src": "Same service + source, recent connections",
    "ct_state_ttl": "State/TTL combination count",
    "ct_dst_ltm": "Connections to same destination (recent)",
    "ct_src_dport_ltm": "Same source to same port (recent)",
    "ct_dst_sport_ltm": "Same destination from same port (recent)",
    "ct_dst_src_ltm": "Same source-destination pair (recent)",
    "is_ftp_login": "FTP login attempted",
    "ct_ftp_cmd": "FTP commands in session",
    "ct_flw_http_mthd": "HTTP GET/POST count",
    "ct_src_ltm": "Connections from same source (recent)",
    "ct_srv_dst": "Same service + destination, recent connections",
    "is_sm_ips_ports": "Same IP and port on both ends",
}


def load_splits():
    """
    Return (train_df, test_df).

    If you downloaded both Kaggle files (training-set and testing-set),
    we use them as given. If only the training-set file is present,
    we hold out 30% of it as a test set ourselves.
    """
    train_df = pd.read_csv(TRAIN_FILE, encoding="utf-8-sig")

    if TEST_FILE.exists():
        test_df = pd.read_csv(TEST_FILE, encoding="utf-8-sig")
    else:
        # stratify keeps the same attack mix in both halves
        train_df, test_df = train_test_split(
            train_df, test_size=0.30, stratify=train_df[TARGET],
            random_state=RANDOM_STATE,
        )

    # Some copies of the dataset have stray spaces in the labels
    for df in (train_df, test_df):
        df[TARGET] = df[TARGET].astype(str).str.strip()

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def make_encoder(train_df):
    """Learn a number for every text value in proto / service / state."""
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    encoder.fit(train_df[CATEGORICAL].astype(str))
    return encoder


def to_features(df, encoder):
    """Turn a raw dataframe into the numeric table the model expects."""
    X = df.drop(columns=[c for c in DROP if c in df.columns]).copy()
    X[CATEGORICAL] = encoder.transform(X[CATEGORICAL].astype(str))
    return X


def nice_name(feature):
    return FEATURE_LABELS.get(feature, feature)


# ---------------------------------------------------------------------------
# MITRE ATT&CK mapping and response playbook
# ---------------------------------------------------------------------------
# MITRE ATT&CK is the industry-standard catalogue of attacker behaviour.
# Mapping our alert types to it tells an analyst what stage of an attack
# they are looking at. The mapping is approximate: UNSW-NB15's categories
# are broad, so where no single technique fits we name only the tactic.
ATTACK_INFO = {
    # UNSW-NB15 categories (replay mode)
    "Reconnaissance": ("TA0043 Reconnaissance", "T1595 Active Scanning",
                       "Rate-limit or block the source IP at the firewall; check which services it probed."),
    "Fuzzers": ("TA0043 Reconnaissance", "T1595.002 Vulnerability Scanning",
                "Block the source IP; review logs of the targeted service for crashes or errors."),
    "Analysis": ("TA0007 Discovery", "T1046 Network Service Discovery",
                 "Block the source IP; confirm no unnecessary ports are exposed."),
    "Exploits": ("TA0001 Initial Access", "T1190 Exploit Public-Facing Application",
                 "Isolate the targeted host; check it is patched; look for new processes or accounts."),
    "DoS": ("TA0040 Impact", "T1498 Network Denial of Service",
            "Enable rate limiting / upstream filtering; block the source if traffic is not spoofed."),
    "Backdoor": ("TA0003 Persistence", "No single technique (varies by backdoor)",
                 "Isolate the host immediately; hunt for unknown services, scheduled tasks and accounts."),
    "Shellcode": ("TA0002 Execution", "T1203 Exploitation for Client Execution",
                  "Isolate the host; capture memory if possible; scan for malware."),
    "Worms": ("TA0008 Lateral Movement", "T1210 Exploitation of Remote Services",
              "Segment the network; isolate infected hosts; patch the exploited service everywhere."),
    "Generic": ("TA0006 Credential Access", "No single technique (attack on block ciphers)",
                "Review cryptographic configuration; rotate keys used by the targeted service."),
    # Live-mode labels
    "Port scan": ("TA0007 Discovery", "T1046 Network Service Discovery",
                  "Block the source IP; confirm only required ports are open on this machine."),
    "Traffic flood": ("TA0040 Impact", "T1498 Network Denial of Service",
                      "Block or rate-limit the source; check whether the machine is still responsive."),
    "Brute force": ("TA0006 Credential Access", "T1110 Brute Force",
                    "Block the source IP; enforce account lockout; check for any successful login."),
    "Unusual traffic": ("Unclassified", "No matching technique",
                        "Investigate manually: compare with the baseline values shown in the explanation."),
}


def attack_info(label):
    """Return (tactic, technique, recommended action) for an alert type."""
    return ATTACK_INFO.get(label, ATTACK_INFO["Unusual traffic"])
