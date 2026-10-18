"""
ui_model.py
The "Model report" tab: how good the replay-mode Random Forest is, how it
compares with other algorithms, and which features matter most.
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from common import ARTIFACTS
from ui_common import BLUE, CLEAN, RED


def model_report(metrics, importance):
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Attacks detected", f"{metrics['detection_rate']:.1%}",
              help="Share of attack flows the model flagged as some kind of attack")
    m2.metric("False alarm rate", f"{metrics['false_alarm_rate']:.1%}",
              help="Share of normal flows wrongly flagged as attacks")
    m3.metric("Attack-type accuracy", f"{metrics['accuracy']:.1%}",
              help="Exact class correct, out of 10 classes")
    m4.metric("Macro F1", f"{metrics['macro_f1']:.2f}",
              help="Average F1 across classes; low because rare classes are hard")
    st.caption(f"Random Forest trained on {metrics['train_rows']:,} UNSW-NB15 flows, "
               f"tested on {metrics['test_rows']:,} held-out flows.")

    # --- algorithm comparison -----------------------------------------------------
    if "comparison" in metrics:
        st.subheader("Why Random Forest? Four algorithms on the same test set")
        comp = pd.DataFrame(metrics["comparison"]).T
        c1, c2 = st.columns([3, 2], gap="large")
        with c1:
            st.dataframe(comp.style.format({
                "accuracy": "{:.1%}", "macro_f1": "{:.2f}", "detection_rate": "{:.1%}",
                "false_alarm_rate": "{:.1%}", "train_seconds": "{:.0f}s"}),
                width="stretch", column_config={
                    "accuracy": "Type accuracy", "macro_f1": "Macro F1",
                    "detection_rate": "Detection", "false_alarm_rate": "False alarms",
                    "train_seconds": "Train time"})
            st.caption("Random Forest ties Gradient Boosting for the lowest false-alarm rate "
                       "(false alarms are what wear analysts out), trains as fast, and SHAP's "
                       "TreeExplainer explains it exactly. The Decision Tree catches slightly more "
                       "attacks but raises far more false alarms.")
        with c2:
            long = comp.reset_index(names="model").melt(
                id_vars="model", value_vars=["detection_rate", "false_alarm_rate"],
                var_name="metric", value_name="value")
            long["metric"] = long.metric.map({"detection_rate": "Detection",
                                              "false_alarm_rate": "False alarms"})
            fig = px.bar(long, x="value", y="model", color="metric", barmode="group",
                         orientation="h", color_discrete_map={"Detection": BLUE, "False alarms": RED})
            fig.update_layout(height=280, xaxis_tickformat=".0%", yaxis_title=None, xaxis_title=None,
                              legend=dict(orientation="h", y=-0.2, title=None),
                              margin=dict(l=0, r=10, t=10, b=0), **CLEAN)
            st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.subheader("Per attack type")
        per = pd.DataFrame(metrics["per_class"]).T[["precision", "recall", "f1-score", "support"]]
        per["support"] = per["support"].astype(int)
        st.dataframe(per.sort_values("support", ascending=False).style.format(
            {"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}", "support": "{:,}"}),
            width="stretch")
        st.caption("DoS, Exploits, Analysis and Backdoor flows look very similar in UNSW-NB15, so "
                   "the model often confuses their type. It still recognises them as attacks, "
                   "which is what the detection rate measures.")
        st.subheader("Most influential features across all alerts")
        top = importance.head(12).iloc[::-1]
        fig = px.bar(top, x="mean_abs_shap", y="label", orientation="h", color_discrete_sequence=[BLUE])
        fig.update_layout(height=380, yaxis_title=None, xaxis_title="Mean |SHAP value|",
                          margin=dict(l=0, r=10, t=10, b=0), **CLEAN)
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.subheader("Confusion matrix")
        st.image(str(ARTIFACTS / "confusion_matrix.png"), width="stretch")
