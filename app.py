import streamlit as st
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import plotly.express as px

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.title("Customer Segmentation Dashboard")

# ---------------------------------------------------
# DATA LAYER
# ---------------------------------------------------
@st.cache_data
def load_data(file):
    if file:
        return pd.read_csv(file)
    else:
        return generate_sample()

def generate_sample():
    np.random.seed(42)
    return pd.DataFrame({
        "Age": np.random.randint(18, 70, 300),
        "Annual_Income": np.random.randint(15, 150, 300),
        "Spending_Score": np.random.randint(1, 100, 300)
    })

def clean_data(df):
    df = df.copy()
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    return df

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.header("Controls")

file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

df = clean_data(load_data(file))

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

features = st.sidebar.multiselect(
    "Features",
    numeric_cols,
    default=numeric_cols[:3]
)

auto_k = st.sidebar.checkbox("Auto-select K (Silhouette)", value=True)

k = st.sidebar.slider("K", 2, 10, 4)

# ---------------------------------------------------
# VALIDATION
# ---------------------------------------------------
if len(features) < 2:
    st.error("Select at least 2 numeric features")
    st.stop()

X = df[features].dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------
# MODEL SELECTION
# ---------------------------------------------------
@st.cache_data
def compute_metrics(X_scaled):
    k_range = range(2, 11)
    wcss, sil = [], []

    for k in k_range:
        model = KMeans(n_clusters=k, n_init=10)
        labels = model.fit_predict(X_scaled)
        wcss.append(model.inertia_)
        sil.append(silhouette_score(X_scaled, labels))

    return list(k_range), wcss, sil

k_range, wcss, sil = compute_metrics(X_scaled)

if auto_k:
    k = k_range[np.argmax(sil)]

# ---------------------------------------------------
# MODEL
# ---------------------------------------------------
model = KMeans(n_clusters=k, n_init=10)
labels = model.fit_predict(X_scaled)

df = df.loc[X.index].copy()
df["Cluster"] = labels

# ---------------------------------------------------
# PCA
# ---------------------------------------------------
pca = PCA(n_components=2)
comp = pca.fit_transform(X_scaled)

df["PC1"] = comp[:, 0]
df["PC2"] = comp[:, 1]

# ---------------------------------------------------
# TABS
# ---------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["Overview", "Model Selection", "Clusters", "Insights"]
)

# ---------------------------------------------------
# OVERVIEW
# ---------------------------------------------------
with tab1:
    st.subheader("Dataset")
    st.dataframe(df.head())

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", len(df))
    col2.metric("Features", len(features))
    col3.metric("Clusters", k)

# ---------------------------------------------------
# MODEL SELECTION
# ---------------------------------------------------
with tab2:
    st.subheader("Elbow Method")

    fig1 = px.line(x=k_range, y=wcss)
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Silhouette Score")

    fig2 = px.line(x=k_range, y=sil)
    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------
# CLUSTERS
# ---------------------------------------------------
with tab3:
    st.subheader("Cluster Visualization")

    fig = px.scatter(
        df, x="PC1", y="PC2",
        color=df["Cluster"].astype(str),
        hover_data=features
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------
# INSIGHTS ENGINE
# ---------------------------------------------------
def label_clusters(profile):
    labels = {}

    for cluster in profile.index:
        row = profile.loc[cluster]
        tags = []

        for col in profile.columns:
            q75 = profile[col].quantile(0.75)
            q25 = profile[col].quantile(0.25)

            if row[col] >= q75:
                tags.append(f"High {col}")
            elif row[col] <= q25:
                tags.append(f"Low {col}")

        labels[cluster] = ", ".join(tags) if tags else "Average"

    return labels

# ---------------------------------------------------
# INSIGHTS
# ---------------------------------------------------
with tab4:
    st.subheader("Cluster Profiles")

    profile = df.groupby("Cluster").mean(numeric_only=True)
    st.dataframe(profile)

    st.subheader("Segment Labels")

    cluster_labels = label_clusters(profile)

    for c, label in cluster_labels.items():
        st.markdown(f"**Cluster {c}: {label}**")
        st.write(profile.loc[c])

    # Distribution
    st.subheader("Cluster Size")

    size_fig = px.histogram(df, x="Cluster")
    st.plotly_chart(size_fig, use_container_width=True)

    # Download
    st.download_button(
        "Download Results",
        df.to_csv(index=False),
        "clusters.csv"
    )
