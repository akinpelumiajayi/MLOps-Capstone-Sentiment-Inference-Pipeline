"""Streamlit front-end for the sentiment API.

Tabs: live inference | batch analysis | live service metrics.
All thresholds come from the API's /info endpoint, so the UI always reflects
how the server actually labels text.
"""

from __future__ import annotations

import os
import time
from typing import Any

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
COLORS = {"positive": "#1a9850", "negative": "#d73027"}
DEMO_TEXTS = [
    "A gripping, beautifully shot film with a career-best performance at its heart.",
    "Two hours of my life I will never get back. Dull, smug and badly paced.",
    "Funny, warm and surprisingly moving; the cast clearly had a great time.",
    "The plot is a mess and the dialogue is painfully clunky.",
    "Not perfect, but ambitious and often thrilling.",
    "It looks expensive, yet it feels completely hollow.",
]

st.set_page_config(page_title="Sentiment MLOps", page_icon=":bar_chart:", layout="wide")


@st.cache_data(ttl=30)
def get_info() -> dict[str, Any] | None:
    try:
        r = requests.get(f"{API_URL}/info", timeout=3)
        return r.json() if r.ok else None
    except requests.RequestException:
        return None


def predict_batch(texts: list[str], batch_size: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        r = requests.post(f"{API_URL}/predict/batch", json={"texts": chunk}, timeout=30)
        r.raise_for_status()
        for text, p in zip(chunk, r.json()["predictions"], strict=True):
            results.append({"text": text, **p})
    return results


info = get_info()

# ---- sidebar -----------------------------------------------------------------
with st.sidebar:
    st.title("Sentiment API")
    if info:
        st.success(f"Online - model `{info['model_version']}`")
        st.markdown(
            f"""
| Setting | Value |
|---|---|
| Model type | `{info.get("model_type")}` |
| Trained at | {info.get("trained_at") or "-"} |
| Validation F1 | {info.get("validation_f1") or 0:.3f} |
| Decision threshold | {info["decision_threshold"]} |
| Low-confidence below | {info["low_confidence_threshold"]} |
| Max batch size | {info["max_batch_size"]} |
"""
        )
    else:
        st.error("API unreachable or model not loaded")
    st.caption(f"API: `{API_URL}`")

tab_live, tab_batch, tab_metrics = st.tabs(["Live inference", "Batch analysis", "Service metrics"])

# ---- live inference ----------------------------------------------------------
with tab_live:
    st.header("Live inference")
    text = st.text_area("Review text", height=140, placeholder="e.g. A gripping, beautifully shot film...")
    if st.button("Predict", type="primary", disabled=info is None):
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            t0 = time.perf_counter()
            resp = requests.post(f"{API_URL}/predict", json={"text": text}, timeout=10)
            latency_ms = (time.perf_counter() - t0) * 1000
            if resp.ok:
                res = resp.json()
                c1, c2, c3 = st.columns(3)
                c1.metric("Sentiment", res["label"].capitalize())
                c2.metric("Confidence", f"{res['confidence']:.1%}")
                c3.metric("Round-trip latency", f"{latency_ms:.0f} ms")
                st.progress(res["positive_probability"], text=f"P(positive) = {res['positive_probability']:.3f}")
                if res["low_confidence"]:
                    st.warning("Low-confidence prediction: consider routing this to human review.")
                st.caption(f"Request ID: `{resp.headers.get('X-Request-ID')}`")
            else:
                st.error(f"API error {resp.status_code}: {resp.text}")

# ---- batch analysis ----------------------------------------------------------
with tab_batch:
    st.header("Batch analysis")
    uploaded = st.file_uploader("CSV with a `text` column", type="csv")
    if uploaded:
        df_in = pd.read_csv(uploaded)
        if "text" not in df_in.columns:
            st.error("CSV must have a `text` column.")
            st.stop()
        texts = df_in["text"].dropna().astype(str).tolist()[:1000]
    else:
        st.info("No file uploaded: using demo reviews.")
        texts = DEMO_TEXTS

    if st.button("Run batch", type="primary", disabled=info is None):
        try:
            rows = predict_batch(texts, int(info["max_batch_size"]) if info else 100)
        except requests.RequestException as exc:
            st.error(f"Batch request failed: {exc}")
            rows = []
        if rows:
            df = pd.DataFrame(rows)
            c1, c2 = st.columns(2)
            counts = df["label"].value_counts().rename_axis("label").reset_index(name="count")
            c1.plotly_chart(
                px.pie(counts, values="count", names="label", color="label", color_discrete_map=COLORS),
                width="stretch",
            )
            c2.plotly_chart(
                px.histogram(df, x="positive_probability", nbins=20, color="label", color_discrete_map=COLORS),
                width="stretch",
            )
            st.metric("Low-confidence share", f"{df['low_confidence'].mean():.1%}")
            st.dataframe(df[["text", "label", "confidence", "low_confidence"]], width="stretch")
            st.download_button("Download results (CSV)", df.to_csv(index=False), "predictions.csv", "text/csv")


# ---- service metrics -----------------------------------------------------------
def parse_metrics(raw: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in raw.splitlines():
        if line and not line.startswith("#"):
            name, _, value = line.rpartition(" ")
            try:
                out[name] = float(value)
            except ValueError:
                continue
    return out


@st.fragment(run_every="15s")
def metrics_panel() -> None:
    try:
        raw = requests.get(f"{API_URL}/metrics", timeout=5).text
    except requests.RequestException as exc:
        st.error(f"Could not fetch metrics: {exc}")
        return
    m = parse_metrics(raw)
    pos = m.get('sentiment_predictions_total{label="positive"}', 0.0)
    neg = m.get('sentiment_predictions_total{label="negative"}', 0.0)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predictions served", int(pos + neg))
    c2.metric("Positive share", f"{pos / (pos + neg):.1%}" if pos + neg else "-")
    c3.metric("Score PSI vs. reference", f"{m.get('sentiment_drift_score_psi', 0.0):.3f}")
    c4.metric("In-flight requests", int(m.get("sentiment_active_requests", 0)))
    st.caption("Refreshes every 15 s from this replica's /metrics. Use Grafana for fleet-wide views.")
    with st.expander("Raw /metrics"):
        st.code(raw, language="text")


with tab_metrics:
    st.header("Service metrics")
    metrics_panel()
