"""
Streamlit dashboard — Sentiment MLOps Capstone
3 tabs: Live Inference | Batch Analysis | Prometheus Metrics
"""
import time
import requests
import pandas as pd
import plotly.express as px
import streamlit as st

API_URL = __import__("os").getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Sentiment MLOps Dashboard",
    page_icon="🤖",
    layout="wide",
)

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    threshold   = st.slider("Confidence threshold", 0.5, 1.0, 0.65, 0.05)
    batch_limit = st.number_input("Batch size limit", 10, 500, 100)
    auto_refresh = st.checkbox("Auto-refresh metrics (30 s)")

    st.markdown("---")
    st.subheader("🏥 API Health")
    try:
        r = requests.get(f"{API_URL}/healthz", timeout=3)
        if r.status_code == 200:
            data = r.json()
            st.success(f"✅ Online  |  model: `{data.get('model_version','?')}`")
        else:
            st.error(f"❌ HTTP {r.status_code}")
    except Exception as e:
        st.error(f"❌ Unreachable\n{e}")

    st.markdown("---")
    st.caption(f"API: `{API_URL}`")

# ── tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Live Inference", "📊 Batch Analysis", "📈 Metrics"])

# ── tab 1: live inference ─────────────────────────────────────────────────────
with tab1:
    st.header("Live Sentiment Inference")
    user_text = st.text_area("Enter review text:", height=150,
                              placeholder="e.g. The model accuracy is outstanding…")

    if st.button("Predict", type="primary"):
        if not user_text.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Calling API…"):
                t0 = time.perf_counter()
                try:
                    resp = requests.post(
                        f"{API_URL}/predict",
                        json={"text": user_text},
                        timeout=10,
                    )
                    latency_ms = (time.perf_counter() - t0) * 1000
                    if resp.status_code == 200:
                        result = resp.json()
                        label = result["label"]
                        conf  = result["confidence"]
                        emoji = "😊" if label == "positive" else "😞"
                        color = "green" if label == "positive" else "red"

                        col1, col2, col3 = st.columns(3)
                        col1.metric("Sentiment", f"{emoji} {label.capitalize()}")
                        col2.metric("Confidence", f"{conf:.1%}")
                        col3.metric("Latency", f"{latency_ms:.1f} ms")

                        st.progress(conf)
                        if conf < threshold:
                            st.warning(
                                f"⚠️ Confidence {conf:.1%} is below your threshold "
                                f"({threshold:.0%}) — treat with caution."
                            )
                    else:
                        st.error(f"API error {resp.status_code}: {resp.text}")
                except Exception as e:
                    st.error(f"Request failed: {e}")

# ── tab 2: batch analysis ─────────────────────────────────────────────────────
with tab2:
    st.header("Batch Sentiment Analysis")
    uploaded = st.file_uploader("Upload CSV with a `text` column", type="csv")

    demo_texts = [
        "This model is surprisingly accurate and fast",
        "Terrible accuracy, completely unreliable results",
        "Works perfectly, highly recommend it",
        "Crashes under load, very unstable deployment",
        "Impressive accuracy on edge cases",
        "Predictions are random at best, totally broken",
    ]

    if uploaded:
        df = pd.read_csv(uploaded).head(batch_limit)
        if "text" not in df.columns:
            st.error("CSV must have a `text` column.")
            st.stop()
        texts = df["text"].tolist()
    else:
        st.info("No file uploaded — using demo texts.")
        texts = demo_texts

    if st.button("Run Batch", type="primary"):
        results = []
        progress = st.progress(0)
        for i, text in enumerate(texts):
            try:
                r = requests.post(f"{API_URL}/predict", json={"text": text}, timeout=10)
                if r.status_code == 200:
                    d = r.json()
                    results.append({
                        "text": text[:80],
                        "label": d["label"],
                        "confidence": d["confidence"],
                    })
            except Exception:
                pass
            progress.progress((i + 1) / len(texts))

        if results:
            df_res = pd.DataFrame(results)
            col1, col2 = st.columns(2)
            with col1:
                counts = df_res["label"].value_counts().reset_index()
                counts.columns = ["label", "count"]
                fig = px.pie(counts, values="count", names="label",
                             title="Label distribution",
                             color="label",
                             color_discrete_map={"positive": "#2ecc71", "negative": "#e74c3c"})
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig2 = px.histogram(df_res, x="confidence", nbins=10,
                                    title="Confidence distribution",
                                    color="label",
                                    color_discrete_map={"positive": "#2ecc71", "negative": "#e74c3c"})
                st.plotly_chart(fig2, use_container_width=True)

            st.dataframe(df_res, use_container_width=True)

# ── tab 3: prometheus metrics ─────────────────────────────────────────────────
with tab3:
    st.header("Prometheus Metrics")

    if auto_refresh:
        time.sleep(30)
        st.rerun()

    try:
        raw = requests.get(f"{API_URL}/metrics", timeout=5).text
        lines = [l for l in raw.splitlines() if not l.startswith("#") and l.strip()]

        def extract(name: str) -> float:
            for line in lines:
                if line.startswith(name + " ") or line.startswith(name + "{"):
                    try:
                        return float(line.split()[-1])
                    except ValueError:
                        pass
            return 0.0

        total_pos = extract('sentiment_requests_total{label="positive"}')
        total_neg = extract('sentiment_requests_total{label="negative"}')
        active    = extract("sentiment_active_requests")
        total     = total_pos + total_neg

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Requests", int(total))
        c2.metric("Positive", int(total_pos))
        c3.metric("Negative", int(total_neg))
        c4.metric("Active Now", int(active))

        if total > 0:
            fig = px.bar(
                x=["Positive", "Negative"],
                y=[total_pos, total_neg],
                title="Request breakdown",
                color=["Positive", "Negative"],
                color_discrete_map={"Positive": "#2ecc71", "Negative": "#e74c3c"},
            )
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Raw /metrics output"):
            st.text(raw)

    except Exception as e:
        st.error(f"Could not fetch metrics: {e}")
