"""
STEP 4 - The dashboard.

    streamlit run app.py

Three tabs:
  Replay demo   one hour of UNSW-NB15 traffic + simulated logins (steps 1-3)
  Live capture  real traffic and real failed logins from live_monitor.py
  Model report  how good the replay model is and why we chose it

The tab contents live in ui_replay.py, ui_live.py and ui_model.py;
shared charts are in ui_common.py.
"""

import json

import pandas as pd
import streamlit as st

from common import ARTIFACTS
from ui_common import SEVERITY_ORDER
from ui_live import live_view
from ui_model import model_report
from ui_replay import replay_view

st.set_page_config(page_title="XAI Cyber Defense Monitor", page_icon="🛡️", layout="wide")


@st.cache_data
def load():
    feed = pd.read_csv(ARTIFACTS / "scored_feed.csv", parse_dates=["timestamp"])
    auth = pd.read_csv(ARTIFACTS / "auth_log.csv", parse_dates=["timestamp"])
    with open(ARTIFACTS / "metrics.json") as f:
        metrics = json.load(f)
    importance = pd.read_csv(ARTIFACTS / "global_importance.csv")
    return feed, auth, metrics, importance


needed = ["scored_feed.csv", "auth_log.csv", "metrics.json", "global_importance.csv"]
missing = [n for n in needed if not (ARTIFACTS / n).exists()]
if missing:
    st.error("The pipeline hasn't been run yet. Run `python run_pipeline.py` first, "
             f"then reload this page. Missing: {', '.join(missing)}")
    st.stop()

feed, auth, metrics, importance = load()

# Replay state: `cursor` = how many flows have "arrived"; `blocked` = ip -> time
state = st.session_state
if "cursor" not in state:
    state.cursor = 400
    state.playing = False
    state.blocked = {}

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Replay controls")
    c1, c2 = st.columns(2)
    if c1.button("Pause" if state.playing else "Play", type="primary", width="stretch"):
        state.playing = not state.playing
        st.rerun()
    if c2.button("Restart", width="stretch"):
        state.cursor, state.blocked = 50, {}
        st.rerun()
    speed = st.select_slider("Flows per second", options=[5, 10, 25, 50, 100], value=25)
    if st.button("Jump to end of hour", width="stretch"):
        state.cursor, state.playing = len(feed), False
        st.rerun()
    shown_sev = st.multiselect("Show alerts of severity", SEVERITY_ORDER[:3],
                               default=SEVERITY_ORDER[:3])
    if state.blocked:
        st.markdown("**Blocked IPs**  \n" + "  \n".join(
            f"`{ip}` at {t:%H:%M:%S}" for ip, t in state.blocked.items()))
        if st.button("Clear blocklist", width="stretch"):
            state.blocked = {}
            st.rerun()

    st.header("Live capture")
    live_refresh = st.toggle("Auto-refresh every 2 s", value=True)

    with st.expander("How the risk score works"):
        st.markdown(
            "Each event gets a score from 0 to 100:\n\n"
            "- **60%** network evidence: the Random Forest's attack probability (replay), or how far "
            "traffic is above this laptop's normal (live)\n"
            "- **25%** failed logins from the same IP in the last 15 min (10+ = full)\n"
            "- **15%** earlier alerts from the same IP in the last 15 min (5+ = full)\n\n"
            "Critical ≥ 80 · High ≥ 60 · Medium ≥ 40. Medium and above raise an alert. "
            "Live mode adds one rule: 10+ failed logins in 15 min is at least High.")

# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.title("XAI Cyber Defense Monitor")
st.caption("Network traffic and login activity, scored together and explained.")

tab_replay, tab_live, tab_model = st.tabs(["Replay demo", "Live capture", "Model report"])
with tab_replay:
    st.caption("Recorded UNSW-NB15 test traffic with a simulated login log, replayed as if live.")
    st.fragment(replay_view, run_every=1 if state.playing else None)(feed, auth, speed, shown_sev)
with tab_live:
    st.caption("Real packets and real failed logins from this laptop, via live_monitor.py.")
    st.fragment(live_view, run_every=2 if live_refresh else None)()
with tab_model:
    model_report(metrics, importance)
