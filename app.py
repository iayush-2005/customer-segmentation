import streamlit as st
import pandas as pd
import numpy as np
import io, json

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer

import plotly.express as px

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# ─────────────────────────────────────
# CONFIG
# ─────────────────────────────────────
st.set_page_config(layout="wide", page_title="Customer Segmentation Pro")

st.title("🚀 Customer Segmentation Pro")
st.caption("K-Means · Auto Clustering · Business Insights")

# ─────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────
file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
else:
    df = pd.read_csv("data/sample.csv")

df.columns = df.columns.str.strip().str.replace(" ", "_")

# ─────────────────────────────────────
# FEATURE SELECTION
# ─────────────────────────────────────
num_cols = df.select_dtypes(include=np.number).columns.tolist()

features = st.multiselect(
    "Select Features",
    num_cols,
    default=num_cols[1:4]
)

# ─────────────────────────────────────
# AUTO K
# ─────────────────────────────────────
def find_best_k(X):
    scores = []
    for k in range(2, 11):
        labels = KMeans(n_clusters=k, n_init=10).fit_predict(X)
        score = silhouette_score(X, labels)
        scores.append(score)
    return np.argmax(scores) + 2, scores

# ─────────────────────────────────────
# RUN
# ─────────────────────────────────────
if st.button("🚀 Run Analysis") and len(features) >= 2:

    X = df[features]
    X = SimpleImputer().fit_transform(X)
    X = StandardScaler().fit_transform(X)

    best_k, scores = find_best_k(X)

    model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = model.fit_predict(X)

    df["Cluster"] = labels

    sil = silhouette_score(X, labels)

    # ─────────────────────────────
    # METRICS DASHBOARD
    # ─────────────────────────────
    st.subheader("📊 Model Performance")

    c1, c2 = st.columns(2)
    c1.metric("Optimal K", best_k)
    c2.metric("Silhouette Score", round(sil, 3))

    # ─────────────────────────────
    # VISUALIZATION
    # ─────────────────────────────
    st.subheader("📈 Cluster Visualization")

    if len(features) >= 3:
        fig = px.scatter_3d(
            df,
            x=features[0],
            y=features[1],
            z=features[2],
            color="Cluster"
        )
    else:
        fig = px.scatter(
            df,
            x=features[0],
            y=features[1],
            color="Cluster"
        )

    st.plotly_chart(fig, use_container_width=True)

    # ─────────────────────────────
    # ELBOW / SILHOUETTE GRAPH
    # ─────────────────────────────
    st.subheader("📉 Silhouette vs K")

    fig2 = px.line(
        x=list(range(2, 11)),
        y=scores,
        markers=True,
        labels={"x": "K", "y": "Silhouette"}
    )

    st.plotly_chart(fig2, use_container_width=True)

    # ─────────────────────────────
    # CLUSTER PROFILE
    # ─────────────────────────────
    st.subheader("📋 Cluster Profiles")

    profile = df.groupby("Cluster").mean(numeric_only=True)
    profile["Count"] = df["Cluster"].value_counts().sort_index()

    st.dataframe(profile)

    # ─────────────────────────────
    # BUSINESS INSIGHTS
    # ─────────────────────────────
    st.subheader("🧠 Business Insights")

    for cid in sorted(df["Cluster"].unique()):
        grp = df[df["Cluster"] == cid]

        st.markdown(f"### Cluster {cid}")

        desc = []
        for f in features:
            val = grp[f].mean()
            desc.append(f"{f}: {val:.2f}")

        st.write(", ".join(desc))

        # Insight logic
        if grp[features[0]].mean() > df[features[0]].median():
            st.success("High-value segment → Focus on premium marketing & retention")
        else:
            st.warning("Low-value segment → Improve engagement & offers")

    # ─────────────────────────────
    # EXPORTS
    # ─────────────────────────────
    st.subheader("📦 Export Data")

    col1, col2, col3 = st.columns(3)

    col1.download_button(
        "CSV",
        df.to_csv(index=False),
        "data.csv"
    )

    # JSON
    col2.download_button(
        "JSON",
        json.dumps(df.to_dict(orient="records")),
        "data.json"
    )

    # PDF
    def generate_pdf():
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer)
        styles = getSampleStyleSheet()

        elements = []
        elements.append(Paragraph("Customer Segmentation Report", styles["Title"]))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(f"Optimal K: {best_k}", styles["Normal"]))
        elements.append(Paragraph(f"Silhouette Score: {sil:.3f}", styles["Normal"]))

        doc.build(elements)
        buffer.seek(0)
        return buffer

    col3.download_button(
        "PDF Report",
        generate_pdf(),
        "report.pdf"
    )
