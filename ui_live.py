"""
ui_live.py
The "Live capture" tab. It does not capture anything itself: it reads the
files that live_monitor.py keeps writing and redraws every 2 seconds.
"""

import json
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from live_monitor import LOGINS_FILE, STATUS_FILE, WINDOWS_FILE
from ui_common import (CLEAN, SEVERITY_COLORS, SEVERITY_ORDER, explanation_chart,
                       incident_report, ip_timeline, response_panel, score_chart,
                       severity_style)

HOW_TO_START = """
**The live monitor isn't running yet.** It captures real traffic on this laptop.

1. Open a terminal **as Administrator** (Windows: right-click Terminal → Run as administrator;
   you also need [Npcap](https://npcap.com/#download) installed).
2. `cd` into the project folder and run:
   ```
   python live_monitor.py --learn 10
   ```
3. For the next 10 minutes use the laptop normally (browse, watch a video) so it learns
   what normal looks like. This tab shows progress.
4. Once it says **DETECTING**, attack it from the second laptop:
   ```
   python attack_demo.py <this laptop's IP> all
   ```
"""


def read_csv(path, **kw):
    try:
        return pd.read_csv(path, **kw) if path.exists() else pd.DataFrame()
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def live_view():
    if not STATUS_FILE.exists():
        st.markdown(HOW_TO_START)
        return
    status = json.loads(STATUS_FILE.read_text())
    age = (datetime.now() - datetime.fromisoformat(status["updated"])).total_seconds()

    # --- monitor status -------------------------------------------------------
    phase = status.get("phase", "?")
    if phase == "stopped" or age > 20:
        st.warning("The live monitor is not running (no update for "
                   f"{age:.0f}s). Showing the last data it recorded. "
                   "Start it again with `python live_monitor.py`.")
    elif phase == "learning":
        left = status.get("seconds_left", 0)
        st.info(f"**Learning normal behaviour** · {status.get('baseline_rows', 0)} traffic samples so "
                f"far · {left // 60}m {left % 60}s left. Use the laptop normally; don't attack yet.")
        return
    else:
        st.success(f"**Detecting** on {status.get('iface')} "
                   f"({', '.join(status.get('local_ips', []))}) · "
                   f"{status.get('packets', 0):,} packets captured · "
                   f"baseline of {status.get('baseline_rows', 0)} samples")
    st.caption(f"Login source: {status.get('login_status', 'unknown')}")

    windows = read_csv(WINDOWS_FILE, parse_dates=["timestamp"])
    logins = read_csv(LOGINS_FILE, parse_dates=["timestamp"])
    if windows.empty:
        st.info("Waiting for the first traffic window ...")
        return
    windows["reasons"] = windows["reasons"].fillna("")
    windows["label"] = windows["label"].fillna("")
    alerts = windows[windows.is_alert].iloc[::-1]

    # --- headline numbers -------------------------------------------------------
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Remote IPs seen", windows.src_ip.nunique())
    k2.metric("Alerts", len(alerts))
    k3.metric("High or Critical", int(alerts.severity.isin(["High", "Critical"]).sum()))
    k4.metric("Failed logins", len(logins))

    # --- activity over the last 10 minutes ---------------------------------------
    recent = windows[windows.timestamp >= windows.timestamp.max() - pd.Timedelta(minutes=10)]
    fig = px.scatter(recent, x="timestamp", y="max_deviation", color="severity",
                     color_discrete_map=SEVERITY_COLORS, category_orders={"severity": SEVERITY_ORDER},
                     hover_data=["src_ip", "pkts_in", "local_ports", "risk"])
    fig.add_hline(y=3, line_dash="dot", line_color="#8A99A8",
                  annotation_text="edge of normal", annotation_position="top left")
    fig.update_layout(height=230, xaxis_title=None, yaxis_title="Spreads above normal",
                      margin=dict(l=0, r=10, t=10, b=0), legend=dict(orientation="h", y=-0.25),
                      **CLEAN)
    st.markdown("**Every remote IP, every window (last 10 minutes)**")
    st.plotly_chart(fig, width="stretch", key="live_activity")

    if alerts.empty:
        st.info("No alerts yet. Everything is within the learned baseline.")
        return

    # --- alerts table ---------------------------------------------------------
    head, dl = st.columns([3, 1])
    head.subheader("Alerts")
    dl.download_button("Export alerts (CSV)", alerts.drop(columns=["reasons"]).to_csv(index=False),
                       file_name="live_alerts.csv", width="stretch")
    if "reputation" not in alerts.columns:        # older capture files
        alerts["reputation"] = ""
    table = alerts[["timestamp", "src_ip", "reputation", "severity", "risk", "label",
                    "max_deviation", "failed_logins_15m", "top_reason"]]
    event = st.dataframe(
        severity_style(table), hide_index=True, height=260, width="stretch",
        on_select="rerun", selection_mode="single-row", key="live_table",
        column_config={
            "timestamp": st.column_config.DatetimeColumn("Time", format="HH:mm:ss"),
            "src_ip": "Source IP", "reputation": "Source reputation", "severity": "Severity",
            "risk": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100, format="%d"),
            "label": "Type (classifier)",
            "max_deviation": st.column_config.NumberColumn("Spreads above normal", format="%.1f"),
            "failed_logins_15m": st.column_config.NumberColumn("Failed logins"),
            "top_reason": "Main reason",
        })

    rows = event.selection.rows if event else []
    picked = alerts.iloc[rows[0]] if rows and rows[0] < len(alerts) else alerts.iloc[0]

    # --- details --------------------------------------------------------------
    st.divider()
    color = SEVERITY_COLORS[picked.severity]
    st.markdown(
        f"### {picked.label} from `{picked.src_ip}`\n"
        f"<span style='color:{color};font-weight:700'>{picked.severity} · risk {picked.risk}</span>"
        f" &nbsp; at {picked.timestamp:%H:%M:%S} · {picked.pkts_in} packets in, "
        f"{picked.local_ports} of our ports touched, {picked.failed_logins_15m} failed logins (15 min).",
        unsafe_allow_html=True)

    key = f"live_{picked.src_ip}_{picked.timestamp:%H%M%S}"
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("**Why it was flagged**")
        if picked.reasons:
            st.plotly_chart(explanation_chart(picked.reasons, "Spreads above this laptop's normal",
                                              with_typical=True), width="stretch", key=f"why_{key}")
            st.caption("Each bar is exactly how far that measurement was above what this laptop "
                       "normally sees. Beyond 3 is unusual; 8 or more counts as fully abnormal.")
        else:
            st.markdown(f"No unusual traffic: flagged because of **{picked.failed_logins_15m} failed "
                        "logins** in 15 minutes (rule: 10 or more is at least High).")
        rep = getattr(picked, "reputation", "")
        if rep:
            st.markdown(f"**Source reputation:** {rep}")
        st.markdown("**Response**")
        response_panel(picked.label, picked.src_ip, key, live=True)
        st.download_button("Incident report",
                           incident_report(picked, picked.label, picked.reasons,
                                           f"live capture on {status.get('iface')}"),
                           file_name=f"incident_{picked.src_ip}_{picked.timestamp:%H%M%S}.md",
                           key=f"report_{key}")
    with b:
        st.markdown("**How the risk score was built**")
        st.plotly_chart(score_chart(picked), width="stretch", key=f"score_{key}")
        st.markdown(f"**Everything from `{picked.src_ip}`**")
        ip_logins = logins[logins.src_ip == picked.src_ip] if not logins.empty else logins
        st.plotly_chart(ip_timeline(windows[windows.src_ip == picked.src_ip], ip_logins,
                                    ip_logins.iloc[0:0]), width="stretch", key=f"ip_{key}")

    if not logins.empty:
        with st.expander(f"Failed logins recorded ({len(logins)})"):
            st.dataframe(logins.iloc[::-1], hide_index=True, width="stretch")
