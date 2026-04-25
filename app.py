import streamlit as st
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import plotly.express as px

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.title("Customer Segmentation using K-Means")

# -------------------------------
# SESSION STATE
# -------------------------------
for key, val in {
    "k": 4,
    "run": False
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# -------------------------------
# DATA LOADING
# -------------------------------
def generate_sample():
    np.random.seed(42)
    return pd.DataFrame({
        "Age": np.random.randint(18, 70, 200),
        "Annual_Income": np.random.randint(15, 150, 200),
        "Spending_Score": np.random.randint(1, 100, 200)
    })

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
else:
    try:
        df = pd.read_csv("data/customer_data.csv")
    except:
        st.warning("No dataset found. Using generated sample data.")
        df = generate_sample()

# Clean columns
df.columns = df.columns.str.strip().str.replace(" ", "_")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# -------------------------------
# FEATURE SELECTION
# -------------------------------
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

if len(numeric_cols) < 2:
    st.error("Need at least 2 numeric columns")
    st.stop()

features = st.multiselect(
    "Select features for clustering",
    numeric_cols,
    default=numeric_cols[:3]
)

# -------------------------------
# K SELECTION
# -------------------------------
k = st.slider("Number of Clusters (K)", 2, 10, st.session_state.k)

if st.button("Run Clustering"):
    st.session_state.run = True

# -------------------------------
# CLUSTERING
# -------------------------------
if st.session_state.run:

    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    # -------------------------------
    # PCA VISUALIZATION
    # -------------------------------
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)

    df["PC1"] = components[:, 0]
    df["PC2"] = components[:, 1]

    st.subheader("Cluster Visualization (PCA)")

    fig = px.scatter(
        df,
        x="PC1",
        y="PC2",
        color=df["Cluster"].astype(str),
        title="Customer Segments"
    )
    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------
    # CLUSTER DISTRIBUTION
    # -------------------------------
    st.subheader("Cluster Distribution")

    dist_fig = px.histogram(df, x="Cluster", color="Cluster")
    st.plotly_chart(dist_fig, use_container_width=True)

    # -------------------------------
    # SAFE AGGREGATION (FIXED)
    # -------------------------------
    st.subheader("Cluster Profile")

    cluster_profile = df.groupby("Cluster").mean(numeric_only=True)
    st.dataframe(cluster_profile)

    # -------------------------------
    # INTERPRETATION
    # -------------------------------
    st.subheader("Insights")

    for i in cluster_profile.index:
        st.write(f"Cluster {i}:")
        st.write(cluster_profile.loc[i])

    # -------------------------------
    # DOWNLOAD
    # -------------------------------
    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Clustered Data",
        csv,
        "clustered_data.csv",
        "text/csv"
    )
