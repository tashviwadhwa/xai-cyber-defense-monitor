"""
ui_common.py
Pieces of the dashboard used by more than one tab: colours, charts,
the MITRE ATT&CK / response panel and the incident report.
"""

import json
from datetime import datetime

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import attack_info

SEVERITY_COLORS = {"Critical": "#C0362C", "High": "#E07B24", "Medium": "#C9A227", "Low": "#8A99A8"}
SEVERITY_ORDER = ["Critical", "High", "Medium", "Low"]
BLUE, PURPLE, GREEN, RED = "#2C5E8A", "#7B4FA3", "#3F8F7A", "#C0362C"
CLEAN = dict(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")


def fmt(v):
    """Short, readable number: 13,500 / 0.25 / 'tcp'."""
    if isinstance(v, str):
        return v
    return f"{v:,.0f}" if abs(v) >= 100 else f"{v:.3g}"


def severity_style(df):
    """Colour the Severity column of a table."""
    return df.style.map(lambda v: f"color: {SEVERITY_COLORS.get(v, '#16283A')}; font-weight: 600",
                        subset=["severity"])


def explanation_chart(reasons_json, x_title, with_typical=False):
    """Horizontal bars: red = pushed towards alert, blue = pushed away."""
    reasons = json.loads(reasons_json)[::-1]            # biggest at the top
    if with_typical:
        labels = [f"{r['label']}: {fmt(r['value'])} (normal ≈ {fmt(r['typical'])})" for r in reasons]
    else:
        labels = [f"{r['label']} = {fmt(r['value'])}" for r in reasons]
    vals = [r["contribution"] for r in reasons]
    fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h",
                           marker_color=[RED if v > 0 else BLUE for v in vals],
                           hovertemplate="%{y}<br>%{x:.2f}<extra></extra>"))
    fig.add_vline(x=0, line_color="#16283A", line_width=1)
    fig.update_layout(height=280, margin=dict(l=0, r=10, t=10, b=30), xaxis_title=x_title, **CLEAN)
    return fig


def score_chart(row):
    """Stacked bar showing where the risk points came from."""
    parts = [("Network", 60 * row.network_score, BLUE),
             ("Failed logins", 25 * row.login_score, PURPLE),
             ("Repeat alerts", 15 * row.repeat_score, GREEN)]
    fig = go.Figure()
    # Live mode's brute-force rule can lift the risk above the three parts
    rule = row.risk - sum(v for _, v, _ in parts)
    if rule >= 1:
        parts.append(("Brute-force rule", rule, "#8A99A8"))
    for name, val, color in parts:
        fig.add_trace(go.Bar(x=[val], y=["Risk"], orientation="h", name=f"{name} ({val:.0f})",
                             marker_color=color, hovertemplate=f"{name}: %{{x:.1f}}<extra></extra>"))
    for cut in (40, 60, 80):
        fig.add_vline(x=cut, line_dash="dot", line_color="#8A99A8")
    fig.update_layout(barmode="stack", height=150, xaxis_range=[0, 100], yaxis_visible=False,
                      legend=dict(orientation="h", y=-0.45), margin=dict(l=0, r=10, t=10, b=0),
                      **CLEAN)
    return fig


def ip_timeline(flows, fails, successes):
    """One IP's activity: its scored traffic (dots) and its logins (markers)."""
    fig = px.scatter(flows, x="timestamp", y="risk", color="severity",
                     color_discrete_map=SEVERITY_COLORS,
                     category_orders={"severity": SEVERITY_ORDER})
    if len(fails):
        fig.add_trace(go.Scatter(x=fails.timestamp, y=[-6] * len(fails), mode="markers",
                                 marker=dict(symbol="x", color=PURPLE, size=7), name="Failed login"))
    if len(successes):
        fig.add_trace(go.Scatter(x=successes.timestamp, y=[-6] * len(successes), mode="markers",
                                 marker=dict(symbol="circle-open", color=GREEN, size=9),
                                 name="Successful login"))
    fig.update_layout(height=260, yaxis_range=[-12, 105], yaxis_title="Risk", xaxis_title=None,
                      margin=dict(l=0, r=10, t=10, b=0), legend=dict(orientation="h", y=-0.2),
                      **CLEAN)
    return fig


def response_panel(label, ip, key, live=False):
    """MITRE ATT&CK mapping, recommended action and (live) a real block command."""
    tactic, technique, action = attack_info(label)
    st.markdown(f"**MITRE ATT&CK:** {tactic} → {technique}  \n**Recommended action:** {action}")
    if live and ip != "local console":
        st.caption("To actually block this IP on the monitored laptop, run in an Administrator terminal:")
        st.code(f'netsh advfirewall firewall add rule name="XAI block {ip}" dir=in action=block remoteip={ip}',
                language="bash", wrap_lines=True)
        st.caption(f'Undo later with: netsh advfirewall firewall delete rule name="XAI block {ip}"')


def incident_report(row, label, reasons_json, source):
    """A short Markdown incident report an analyst could file or email."""
    tactic, technique, action = attack_info(label)
    lines = [
        f"# Incident report: {label} from {row.src_ip}",
        "",
        f"- Generated: {datetime.now():%Y-%m-%d %H:%M}",
        f"- Data source: {source}",
        f"- Detected at: {row.timestamp:%Y-%m-%d %H:%M:%S}",
        f"- Severity: {row.severity} (risk {row.risk}/100)",
        f"- Failed logins from this source (15 min): {row.failed_logins_15m}",
        f"- Earlier alerts from this source (15 min): {row.prior_alerts_15m}",
        "",
        "## Risk breakdown",
        f"- Network evidence: {60 * row.network_score:.0f} / 60",
        f"- Failed logins: {25 * row.login_score:.0f} / 25",
        f"- Repeat behaviour: {15 * row.repeat_score:.0f} / 15",
        "",
        "## Why it was flagged",
    ]
    if reasons_json:
        for r in json.loads(reasons_json):
            lines.append(f"- {r['label']} = {fmt(r['value'])} (effect {r['contribution']:+.2f})")
    else:
        lines.append("- Flagged on failed logins only.")
    lines += ["", "## MITRE ATT&CK", f"- Tactic: {tactic}", f"- Technique: {technique}",
              "", "## Recommended action", f"- {action}", ""]
    return "\n".join(lines)
