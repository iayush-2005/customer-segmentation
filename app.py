import streamlit as st
import pandas as pd
import numpy as np
import io, time, json

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import plotly.express as px

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="Customer Segmentation Pro", layout="wide")

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
for key, val in {
    "df_raw": None,
    "features": [],
    "k": 5,
    "run": False
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    return df

def run_pipeline(df, features, k):
    X = df[features].values

    X = SimpleImputer().fit_transform(X)
    X = StandardScaler().fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    df_out = df.copy()
    df_out["Cluster"] = labels

    sil = silhouette_score(X, labels)
    db = davies_bouldin_score(X, labels)
    ch = calinski_harabasz_score(X, labels)

    if len(features) >= 3:
        fig3d = px.scatter_3d(
            df_out,
            x=features[0], y=features[1], z=features[2],
            color=df_out["Cluster"].astype(str)
        )
    else:
        fig3d = None

    return df_out, sil, db, ch, fig3d

def generate_pdf(df, sil, db, ch):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    elements = []
    elements.append(Paragraph("Customer Segmentation Report", styles["Title"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph(f"Silhouette: {sil:.3f}", styles["Normal"]))
    elements.append(Paragraph(f"Davies-Bouldin: {db:.3f}", styles["Normal"]))
    elements.append(Paragraph(f"Calinski-Harabasz: {ch:.3f}", styles["Normal"]))

    doc.build(elements)
    buffer.seek(0)
    return buffer

def generate_excel(df):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name="Clustered_Data", index=False)
        df.groupby("Cluster").mean().to_excel(writer, sheet_name="Cluster_Profile")
    buffer.seek(0)
    return buffer

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────

st.title("🚀 Customer Segmentation Pro")

# STEP 1
st.subheader("Step 1: Upload CSV")
file = st.file_uploader("Upload dataset", type=["csv"])

if file:
    st.session_state.df_raw = load_data(file)

# STEP 2
if st.session_state.df_raw is not None:
    df = st.session_state.df_raw
    st.subheader("Step 2: Select Features")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    st.session_state.features = st.multiselect(
        "Select numeric features",
        numeric_cols,
        default=st.session_state.features
    )

# STEP 3
if len(st.session_state.features) >= 2:
    st.subheader("Step 3: Choose K")
    st.session_state.k = st.slider("Clusters", 2, 10, st.session_state.k)

# STEP 4
if len(st.session_state.features) >= 2:
    if st.button("Run Clustering"):
        st.session_state.run = True

# ─────────────────────────────────────────────
# EXECUTION
# ─────────────────────────────────────────────
if st.session_state.run:
    df_out, sil, db, ch, fig3d = run_pipeline(
        st.session_state.df_raw,
        st.session_state.features,
        st.session_state.k
    )

    st.success("Clustering Complete")

    # METRICS
    col1, col2, col3 = st.columns(3)
    col1.metric("Silhouette", round(sil, 3))
    col2.metric("Davies-Bouldin", round(db, 3))
    col3.metric("Calinski-Harabasz", round(ch, 1))

    # 3D Plot
    if fig3d:
        st.plotly_chart(fig3d, use_container_width=True)

    # DATA
    st.subheader("Clustered Data")
    st.dataframe(df_out)

    # EXPORTS
    st.subheader("Exports")

    col1, col2, col3, col4 = st.columns(4)

    # CSV
    col1.download_button(
        "CSV",
        df_out.to_csv(index=False),
        "data.csv"
    )

    # EXCEL
    col2.download_button(
        "Excel",
        generate_excel(df_out),
        "data.xlsx"
    )

    # JSON
    col3.download_button(
        "JSON",
        json.dumps(df_out.to_dict(orient="records")),
        "data.json"
    )

    # PDF
    col4.download_button(
        "PDF",
        generate_pdf(df_out, sil, db, ch),
        "report.pdf"
    )
