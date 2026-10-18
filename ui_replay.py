"""
ui_replay.py
The "Replay demo" tab: one hour of UNSW-NB15 test traffic plus a simulated
login log, revealed a little at a time as if it were arriving live.
All scoring was done in advance by step 3; this tab only displays it.
"""

import plotly.express as px
import streamlit as st

from ui_common import (BLUE, CLEAN, SEVERITY_COLORS, explanation_chart, incident_report,
                       ip_timeline, response_panel, score_chart, severity_style)


def replay_view(feed, auth, speed, shown_sev):
    state = st.session_state
    if state.playing:
        state.cursor = min(state.cursor + speed, len(feed))
        if state.cursor >= len(feed):
            state.playing = False
            st.rerun(scope="app")          # refresh the sidebar's Play button

    seen = feed.iloc[:state.cursor]
    now = seen.timestamp.max()
    alerts = seen[seen.is_alert].copy()

    # An alert from an IP after the analyst blocked it would have been
    # dropped by the firewall: mark it "Blocked" instead of "Open".
    alerts["status"] = "Open"
    for ip, when in state.blocked.items():
        alerts.loc[(alerts.src_ip == ip) & (alerts.timestamp > when), "status"] = "Blocked"
    open_alerts = alerts[alerts.status == "Open"]
    shown = alerts[alerts.severity.isin(shown_sev)].iloc[::-1]          # newest first

    # --- headline numbers ---------------------------------------------------
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Flows analysed", f"{len(seen):,}")
    k2.metric("Open alerts", f"{len(open_alerts):,}")
    k3.metric("Critical", f"{(open_alerts.severity == 'Critical').sum():,}")
    k4.metric("Suspicious IPs",
              f"{open_alerts[open_alerts.severity.isin(['Critical', 'High'])].src_ip.nunique()}")
    k5.metric("Blocked IPs", len(state.blocked))
    st.progress(state.cursor / len(feed),
                text=f"Replay clock {now:%H:%M:%S} · {state.cursor:,} of {len(feed):,} flows")

    left, right = st.columns([2, 1], gap="large")

    # --- alerts table -----------------------------------------------------------
    with left:
        head, dl = st.columns([3, 1])
        head.subheader("Alerts")
        dl.download_button("Export alerts (CSV)",
                           shown.drop(columns=["reasons"]).to_csv(index=False),
                           file_name="replay_alerts.csv", width="stretch")
        table = shown[["flow_id", "timestamp", "src_ip", "status", "severity", "risk",
                       "pred_cat", "attack_prob", "failed_logins_15m", "top_reason"]]
        event = st.dataframe(
            severity_style(table), hide_index=True, height=360, width="stretch",
            on_select="rerun", selection_mode="single-row", key="replay_table",
            column_config={
                "flow_id": st.column_config.NumberColumn("Flow", format="%d"),
                "timestamp": st.column_config.DatetimeColumn("Time", format="HH:mm:ss"),
                "src_ip": "Source IP", "status": "Status", "severity": "Severity",
                "risk": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100, format="%d"),
                "pred_cat": "Detected as",
                "attack_prob": st.column_config.NumberColumn("Attack prob.", format="%.2f"),
                "failed_logins_15m": st.column_config.NumberColumn("Failed logins"),
                "top_reason": "Main reason",
            },
        )
        if state.playing:
            st.caption("Pause the replay to click an alert. While playing, the details below follow "
                       "the highest-risk recent alert.")

    # --- riskiest IPs and attack mix -----------------------------------------
    with right:
        st.subheader("Riskiest source IPs")
        if len(open_alerts):
            ips = (open_alerts.groupby("src_ip")
                   .agg(alerts=("flow_id", "count"), max_risk=("risk", "max"))
                   .sort_values(["max_risk", "alerts"], ascending=False).head(8).reset_index())
            fails_now = auth[(auth.outcome == "FAILED") & (auth.timestamp <= now)]
            ips["failed"] = ips.src_ip.map(fails_now.src_ip.value_counts()).fillna(0).astype(int)
            st.dataframe(ips, hide_index=True, width="stretch", column_config={
                "src_ip": "IP", "alerts": "Alerts",
                "max_risk": st.column_config.NumberColumn("Peak risk"),
                "failed": st.column_config.NumberColumn("Failed logins")})
            mix = open_alerts.pred_cat.value_counts().reset_index()
            fig = px.bar(mix, x="count", y="pred_cat", orientation="h", color_discrete_sequence=[BLUE])
            fig.update_layout(height=250, yaxis_title=None, xaxis_title="Open alerts by attack type",
                              yaxis=dict(categoryorder="total ascending"),
                              margin=dict(l=0, r=10, t=10, b=0), **CLEAN)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No open alerts. Press Play to start the replay.")

    # --- details for one alert --------------------------------------------------
    if len(shown) == 0:
        return
    picked = None
    rows = event.selection.rows if event and not state.playing else []
    if rows and rows[0] < len(table):
        picked = shown.iloc[rows[0]]
    if picked is None:                               # default: riskiest recent alert
        recent = shown.head(50)
        picked = recent.loc[recent.risk.idxmax()]

    st.divider()
    color = SEVERITY_COLORS[picked.severity]
    st.markdown(
        f"### Flow {picked.flow_id} from `{picked.src_ip}`\n"
        f"<span style='color:{color};font-weight:700'>{picked.severity} · risk {picked.risk}</span>"
        f" &nbsp; at {picked.timestamp:%H:%M:%S}. Attack probability **{picked.attack_prob:.0%}**, "
        f"most likely type **{picked.pred_cat}** ({picked.confidence:.0%}). "
        f"Actual label in dataset: **{picked.true_cat}**.",
        unsafe_allow_html=True)

    key = f"{picked.flow_id}_{state.cursor}"
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown(f"**Why the model thinks this looks like {picked.explained_class}**")
        st.plotly_chart(explanation_chart(picked.reasons, "SHAP value (effect on the model's score)"),
                        width="stretch", key=f"shap_{key}")
        st.caption("Red bars pushed the model towards this attack type, blue bars pushed it away.")
        st.markdown("**Response**")
        response_panel(picked.explained_class, picked.src_ip, key)
        c1, c2 = st.columns(2)
        if picked.src_ip in state.blocked:
            c1.button(f"Blocked at {state.blocked[picked.src_ip]:%H:%M:%S}", disabled=True,
                      width="stretch", key=f"blocked_{key}")
        elif c1.button(f"Block {picked.src_ip}", type="primary", width="stretch", key=f"block_{key}"):
            state.blocked[picked.src_ip] = now
            st.rerun(scope="app")
        c2.download_button("Incident report",
                           incident_report(picked, picked.explained_class, picked.reasons,
                                           "UNSW-NB15 replay + simulated login log"),
                           file_name=f"incident_flow_{picked.flow_id}.md", width="stretch",
                           key=f"report_{key}")
    with b:
        st.markdown("**How the risk score was built**")
        st.plotly_chart(score_chart(picked), width="stretch", key=f"score_{key}")
        st.markdown(f"**Everything from `{picked.src_ip}` so far**")
        ip_auth = auth[(auth.src_ip == picked.src_ip) & (auth.timestamp <= now)]
        st.plotly_chart(ip_timeline(seen[seen.src_ip == picked.src_ip],
                                    ip_auth[ip_auth.outcome == "FAILED"],
                                    ip_auth[ip_auth.outcome == "SUCCESS"]),
                        width="stretch", key=f"ip_{key}")
