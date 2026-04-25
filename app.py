import streamlit as st
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import plotly.express as px

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(layout="wide")
st.title("Customer Segmentation Dashboard")

# -----------------------------------
# DATA LOADING
# -----------------------------------
def generate_sample():
    np.random.seed(42)
    return pd.DataFrame({
        "Age": np.random.randint(18, 70, 300),
        "Annual_Income": np.random.randint(15, 150, 300),
        "Spending_Score": np.random.randint(1, 100, 300)
    })

file = st.file_uploader("Upload Dataset", type=["csv"])

if file:
    df = pd.read_csv(file)
else:
    try:
        df = pd.read_csv("data/customer_data.csv")
    except:
        df = generate_sample()
        st.info("Using generated dataset")

df.columns = df.columns.str.strip().str.replace(" ", "_")

# -----------------------------------
# FEATURE SELECTION
# -----------------------------------
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

features = st.multiselect(
    "Select Features",
    numeric_cols,
    default=numeric_cols[:3]
)

if len(features) < 2:
    st.warning("Select at least 2 features")
    st.stop()

X = df[features]

# -----------------------------------
# SCALING
# -----------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------------
# MODEL SELECTION
# -----------------------------------
col1, col2 = st.columns(2)

with col1:
    wcss = []
    k_range = range(2, 11)

    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10)
        km.fit(X_scaled)
        wcss.append(km.inertia_)

    fig_elbow = px.line(x=list(k_range), y=wcss,
                        title="Elbow Method",
                        markers=True)
    st.plotly_chart(fig_elbow, use_container_width=True)

with col2:
    sil_scores = []

    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil_scores.append(silhouette_score(X_scaled, labels))

    fig_sil = px.line(x=list(k_range), y=sil_scores,
                      title="Silhouette Score",
                      markers=True)
    st.plotly_chart(fig_sil, use_container_width=True)

# -----------------------------------
# K SELECTION
# -----------------------------------
k = st.slider("Number of Clusters (K)", 2, 10, 4)

# -----------------------------------
# CLUSTERING
# -----------------------------------
kmeans = KMeans(n_clusters=k, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# -----------------------------------
# KPI ROW
# -----------------------------------
c1, c2, c3 = st.columns(3)

c1.metric("Total Customers", len(df))
c2.metric("Features Used", len(features))
c3.metric("Clusters", k)

# -----------------------------------
# PCA VISUALIZATION
# -----------------------------------
pca = PCA(n_components=2)
comp = pca.fit_transform(X_scaled)

df["PC1"] = comp[:, 0]
df["PC2"] = comp[:, 1]

st.subheader("Cluster Visualization")

fig_scatter = px.scatter(
    df,
    x="PC1",
    y="PC2",
    color=df["Cluster"].astype(str),
    opacity=0.7
)
st.plotly_chart(fig_scatter, use_container_width=True)

# -----------------------------------
# CLUSTER SIZE
# -----------------------------------
st.subheader("Cluster Distribution")

cluster_counts = df["Cluster"].value_counts().reset_index()
cluster_counts.columns = ["Cluster", "Count"]

fig_bar = px.bar(cluster_counts, x="Cluster", y="Count")
st.plotly_chart(fig_bar, use_container_width=True)

# -----------------------------------
# FEATURE DISTRIBUTION (KEY UPGRADE)
# -----------------------------------
st.subheader("Feature Distribution per Cluster")

for feature in features:
    fig_box = px.box(df, x="Cluster", y=feature, points="outliers")
    st.plotly_chart(fig_box, use_container_width=True)

# -----------------------------------
# CLUSTER PROFILE
# -----------------------------------
st.subheader("Cluster Profiles")

profile = df.groupby("Cluster").mean(numeric_only=True)
st.dataframe(profile)

# -----------------------------------
# SMART INSIGHTS
# -----------------------------------
st.subheader("Insights")

def label_cluster(row):
    labels = []

    for col in features:
        if row[col] > profile[col].mean():
            labels.append(f"High {col}")
        else:
            labels.append(f"Low {col}")

    return ", ".join(labels)

for i in profile.index:
    st.markdown(f"### Cluster {i}")
    st.write(label_cluster(profile.loc[i]))
    st.write(profile.loc[i])

# -----------------------------------
# DOWNLOAD
# -----------------------------------
csv = df.to_csv(index=False).encode()
st.download_button("Download Results", csv, "clusters.csv")
