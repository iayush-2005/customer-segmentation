import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import io
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #00d4ff;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 0.95rem;
        color: #64748b;
        margin-bottom: 1.5rem;
        font-family: monospace;
    }
    .metric-card {
        background: #0e1420;
        border: 1px solid #1e2d4a;
        border-left: 3px solid #00d4ff;
        padding: 14px 18px;
        border-radius: 4px;
        margin-bottom: 8px;
    }
    .cluster-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 3px;
        font-size: 0.75rem;
        font-weight: 600;
        font-family: monospace;
    }
    .section-divider {
        border: none;
        border-top: 1px solid #1e2d4a;
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

PALETTE = ['#00d4ff', '#7c3aed', '#10b981', '#f59e0b', '#ef4444',
           '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#a855f7']

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_uploaded_dataset(file_bytes, file_name):
    """Cache by file content hash — avoids reprocessing same file."""
    import io as _io
    df = pd.read_csv(_io.BytesIO(file_bytes))
    df.columns = df.columns.str.strip().str.replace(' ', '_')
    return df


@st.cache_data(show_spinner=False)
def run_kmeans_pipeline(data_key, feature_cols_tuple, k):
    """
    Cached clustering pipeline.
    data_key is a hashable identifier; features passed as tuple.
    """
    return None  # placeholder — see actual call below


def run_pipeline(df, feature_cols, k):
    X = df[list(feature_cols)].values
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow + Silhouette
    wcss, sil = [], []
    for ki in range(2, 11):
        km = KMeans(n_clusters=ki, init='k-means++', n_init=10, random_state=42)
        lbl = km.fit_predict(X_scaled)
        wcss.append(km.inertia_)
        sil.append(silhouette_score(X_scaled, lbl))

    # Final model
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    sil_final = silhouette_score(X_scaled, labels)

    centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
    centers_df = pd.DataFrame(centers_original, columns=list(feature_cols))
    centers_df['Cluster'] = range(k)

    f1, f2 = list(feature_cols)[0], list(feature_cols)[-1]
    labels_map = {}
    for i, row in centers_df.iterrows():
        v1 = 'High' if row[f1] > centers_df[f1].median() else 'Low'
        v2 = 'High' if row[f2] > centers_df[f2].median() else 'Low'
        labels_map[i] = f'{v1} {f1.replace("_"," ")} / {v2} {f2.replace("_"," ")}'

    result_df = df.copy()
    result_df['Cluster'] = labels
    result_df['Segment'] = result_df['Cluster'].map(labels_map)

    # PCA for 2D vis if >2 features
    if len(feature_cols) > 2:
        pca = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(X_scaled)
        ev = pca.explained_variance_ratio_.sum() * 100
    else:
        X_2d = X_scaled
        ev = 100.0

    return {
        'df': result_df,
        'X_scaled': X_scaled,
        'X_2d': X_2d,
        'ev': ev,
        'wcss': wcss,
        'sil': sil,
        'sil_final': sil_final,
        'centers_df': centers_df,
        'labels_map': labels_map,
        'inertia': kmeans.inertia_,
        'feature_cols': list(feature_cols),
    }


def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='#0e1420')
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎯 Configuration")
    st.markdown("---")

    st.markdown("**Upload Dataset**")
    st.caption("CSV with at least 2 numeric columns. Recommended: Mall Customers dataset.")

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    df_raw = None
    if uploaded is not None:
        try:
            file_bytes = uploaded.read()
            df_raw = load_uploaded_dataset(file_bytes, uploaded.name)
            st.success(f"✓ {uploaded.name}")
            st.caption(f"{df_raw.shape[0]} rows · {df_raw.shape[1]} columns")
        except Exception as e:
            st.error(f"Failed to read file: {e}")
    else:
        st.info("↑ Upload a CSV to begin.")
        st.markdown("---")
        st.markdown("**Need a dataset?**")
        st.caption("Download Mall Customers CSV:")
        st.code("kaggle datasets download\n-d vjchoudhary7/\ncustomer-segmentation-tutorial", language="text")

    if df_raw is not None:
        st.markdown("---")
        # Feature selection
        st.markdown("**Clustering Features**")
        numeric_cols = df_raw.select_dtypes(include=np.number).columns.tolist()
        id_like = [c for c in numeric_cols
                   if any(x in c.lower() for x in ['id', 'index', 'no'])]
        default_feats = [c for c in numeric_cols if c not in id_like][-2:]

        selected_features = st.multiselect(
            "Select 2+ numeric features",
            options=[c for c in numeric_cols if c not in id_like],
            default=default_feats
        )

        st.markdown("---")
        # K selection
        st.markdown("**Number of Clusters (K)**")
        k_value = st.slider("K", min_value=2, max_value=10, value=5, step=1)

        st.markdown("---")
        st.markdown("**About**")
        st.caption("18B1WCI675 · DMDW Lab\nJUIT Waknaghat")


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">🎯 Customer Segmentation</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">K-Means Clustering · Interactive Analysis · DMDW Lab Project</div>', unsafe_allow_html=True)

if df_raw is None:
    st.info("👈 Configure your dataset in the sidebar to get started.")
    st.stop()

if not selected_features or len(selected_features) < 2:
    st.warning("Select at least 2 features from the sidebar to run clustering.")
    st.stop()

# Run pipeline
with st.spinner("Running K-Means pipeline..."):
    results = run_pipeline(df_raw, tuple(selected_features), k_value)

df_out = results['df']
labels_map = results['labels_map']

# ── METRICS ROW ───────────────────────────────
st.markdown("### 📊 Model Metrics")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Clusters (K)", k_value)
m2.metric("Silhouette Score", f"{results['sil_final']:.4f}")
m3.metric("WCSS (Inertia)", f"{results['inertia']:.1f}")
m4.metric("Total Customers", len(df_out))

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ── TABS ─────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Cluster Visualizations",
    "📉 Elbow & Silhouette",
    "🔍 EDA",
    "📋 Data & Export"
])

# ── TAB 1: CLUSTER VIZ ───────────────────────
with tab1:
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("#### Cluster Scatter Plot")
        fig, ax = plt.subplots(figsize=(8, 5.5))
        fig.patch.set_facecolor('#0e1420')
        ax.set_facecolor('#080c14')

        f1, f2 = selected_features[0], selected_features[-1]
        for cid in sorted(df_out['Cluster'].unique()):
            mask = df_out['Cluster'] == cid
            ax.scatter(
                df_out.loc[mask, f1], df_out.loc[mask, f2],
                s=65, alpha=0.85,
                color=PALETTE[cid % len(PALETTE)],
                label=f'C{cid}: {labels_map[cid][:28]}'
            )
        cx = results['centers_df'][f1]
        cy = results['centers_df'][f2]
        ax.scatter(cx, cy, s=280, marker='X', c='white',
                   edgecolors='black', linewidths=1.5, zorder=5, label='Centroids')

        ax.set_xlabel(f1.replace('_', ' '), color='#94a3b8')
        ax.set_ylabel(f2.replace('_', ' '), color='#94a3b8')
        ax.tick_params(colors='#64748b')
        ax.spines[:].set_color('#1e2d4a')
        ax.legend(fontsize=7.5, facecolor='#0e1420', edgecolor='#1e2d4a',
                  labelcolor='white', loc='best')
        ax.grid(alpha=0.15, color='#1e2d4a')
        ax.set_title(f'K-Means Clusters (K={k_value})', color='white', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        st.download_button("⬇ Download Plot", fig_to_bytes(fig),
                           "cluster_scatter.png", "image/png")
        plt.close()

    with col2:
        st.markdown("#### Cluster Sizes")
        fig2, ax2 = plt.subplots(figsize=(5, 5.5))
        fig2.patch.set_facecolor('#0e1420')
        ax2.set_facecolor('#080c14')

        counts = df_out['Cluster'].value_counts().sort_index()
        bars = ax2.bar(
            [f'C{i}' for i in counts.index],
            counts.values,
            color=[PALETTE[i % len(PALETTE)] for i in counts.index],
            edgecolor='white', alpha=0.9, width=0.6
        )
        for bar, val in zip(bars, counts.values):
            ax2.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.5,
                     str(val), ha='center', va='bottom',
                     color='white', fontweight='bold', fontsize=10)

        ax2.set_xlabel('Cluster', color='#94a3b8')
        ax2.set_ylabel('Count', color='#94a3b8')
        ax2.tick_params(colors='#64748b')
        ax2.spines[:].set_color('#1e2d4a')
        ax2.grid(axis='y', alpha=0.15, color='#1e2d4a')
        ax2.set_title('Customers per Cluster', color='white', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    # PCA if >2 features
    if len(selected_features) > 2:
        st.markdown(f"#### PCA 2D Projection ({results['ev']:.1f}% variance explained)")
        fig3, ax3 = plt.subplots(figsize=(8, 4.5))
        fig3.patch.set_facecolor('#0e1420')
        ax3.set_facecolor('#080c14')
        X_2d = results['X_2d']
        labels_arr = df_out['Cluster'].values
        for cid in sorted(df_out['Cluster'].unique()):
            mask = labels_arr == cid
            ax3.scatter(X_2d[mask, 0], X_2d[mask, 1], s=55, alpha=0.8,
                        color=PALETTE[cid % len(PALETTE)], label=f'C{cid}')
        ax3.set_xlabel('PC1', color='#94a3b8')
        ax3.set_ylabel('PC2', color='#94a3b8')
        ax3.tick_params(colors='#64748b')
        ax3.spines[:].set_color('#1e2d4a')
        ax3.legend(facecolor='#0e1420', edgecolor='#1e2d4a', labelcolor='white')
        ax3.grid(alpha=0.15, color='#1e2d4a')
        ax3.set_title('PCA Projection of Clusters', color='white', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

    # Boxplots
    st.markdown("#### Feature Distribution by Cluster")
    ncols = min(len(selected_features), 3)
    fig4, axes4 = plt.subplots(1, ncols, figsize=(5 * ncols, 4))
    fig4.patch.set_facecolor('#0e1420')
    if ncols == 1:
        axes4 = [axes4]
    for ax4, feat in zip(axes4, selected_features[:ncols]):
        data_list = [df_out[df_out['Cluster'] == c][feat].values
                     for c in sorted(df_out['Cluster'].unique())]
        bp = ax4.boxplot(data_list, patch_artist=True)
        for patch, color in zip(bp['boxes'], PALETTE):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color='white')
        ax4.set_facecolor('#080c14')
        ax4.set_title(feat.replace('_', ' '), color='white', fontweight='bold')
        ax4.set_xticklabels([f'C{i}' for i in sorted(df_out['Cluster'].unique())],
                             color='#64748b')
        ax4.tick_params(colors='#64748b')
        ax4.spines[:].set_color('#1e2d4a')
        ax4.grid(axis='y', alpha=0.15, color='#1e2d4a')
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()


# ── TAB 2: ELBOW ─────────────────────────────
with tab2:
    col_e1, col_e2 = st.columns(2)

    with col_e1:
        st.markdown("#### Elbow Curve (WCSS vs K)")
        fig5, ax5 = plt.subplots(figsize=(6.5, 4))
        fig5.patch.set_facecolor('#0e1420')
        ax5.set_facecolor('#080c14')
        k_range = list(range(2, 11))
        ax5.plot(k_range, results['wcss'], 'o-', color='#00d4ff',
                 linewidth=2.5, markersize=8)
        ax5.fill_between(k_range, results['wcss'], alpha=0.08, color='#00d4ff')
        ax5.axvline(x=k_value, color='#ef4444', linestyle='--',
                    alpha=0.8, label=f'Selected K={k_value}')
        ax5.set_xlabel('K', color='#94a3b8')
        ax5.set_ylabel('WCSS', color='#94a3b8')
        ax5.tick_params(colors='#64748b')
        ax5.spines[:].set_color('#1e2d4a')
        ax5.grid(alpha=0.15, color='#1e2d4a')
        ax5.legend(facecolor='#0e1420', edgecolor='#1e2d4a', labelcolor='white')
        ax5.set_title('Elbow Method', color='white', fontweight='bold')
        ax5.set_xticks(k_range)
        plt.tight_layout()
        st.pyplot(fig5)
        plt.close()

    with col_e2:
        st.markdown("#### Silhouette Score vs K")
        fig6, ax6 = plt.subplots(figsize=(6.5, 4))
        fig6.patch.set_facecolor('#0e1420')
        ax6.set_facecolor('#080c14')
        best_k_idx = results['sil'].index(max(results['sil']))
        best_k = k_range[best_k_idx]
        ax6.plot(k_range, results['sil'], 's-', color='#10b981',
                 linewidth=2.5, markersize=8)
        ax6.fill_between(k_range, results['sil'], alpha=0.08, color='#10b981')
        ax6.axvline(x=best_k, color='#f59e0b', linestyle='--',
                    alpha=0.8, label=f'Best K={best_k}')
        ax6.axvline(x=k_value, color='#ef4444', linestyle='--',
                    alpha=0.8, label=f'Selected K={k_value}')
        ax6.set_xlabel('K', color='#94a3b8')
        ax6.set_ylabel('Silhouette Score', color='#94a3b8')
        ax6.tick_params(colors='#64748b')
        ax6.spines[:].set_color('#1e2d4a')
        ax6.grid(alpha=0.15, color='#1e2d4a')
        ax6.legend(facecolor='#0e1420', edgecolor='#1e2d4a', labelcolor='white')
        ax6.set_title('Silhouette Score', color='white', fontweight='bold')
        ax6.set_xticks(k_range)
        plt.tight_layout()
        st.pyplot(fig6)
        plt.close()

    st.markdown("#### Scores Table")
    score_df = pd.DataFrame({
        'K': k_range,
        'WCSS': [f'{w:.2f}' for w in results['wcss']],
        'Silhouette Score': [f'{s:.4f}' for s in results['sil']]
    })
    score_df['Best'] = score_df['K'].apply(
        lambda x: '⭐ Best Silhouette' if x == best_k else
                  ('✅ Selected' if x == k_value else ''))
    st.dataframe(score_df, use_container_width=True, hide_index=True)


# ── TAB 3: EDA ───────────────────────────────
with tab3:
    st.markdown("#### Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df_raw.shape[0])
    c2.metric("Columns", df_raw.shape[1])
    c3.metric("Null Values", int(df_raw.isnull().sum().sum()))

    st.dataframe(df_raw.describe().round(2), use_container_width=True)

    st.markdown("#### Feature Distributions")
    num_cols = df_raw.select_dtypes(include=np.number).columns.tolist()
    ncols2 = min(len(num_cols), 3)
    fig7, axes7 = plt.subplots(1, ncols2, figsize=(5.5 * ncols2, 4))
    fig7.patch.set_facecolor('#0e1420')
    if ncols2 == 1:
        axes7 = [axes7]
    for ax7, col, color in zip(axes7, num_cols[:ncols2], PALETTE):
        ax7.hist(df_raw[col].dropna(), bins=20, color=color,
                 edgecolor='white', alpha=0.85)
        ax7.set_facecolor('#080c14')
        ax7.set_title(col.replace('_', ' '), color='white', fontweight='bold')
        ax7.tick_params(colors='#64748b')
        ax7.spines[:].set_color('#1e2d4a')
        ax7.grid(axis='y', alpha=0.15, color='#1e2d4a')
    plt.tight_layout()
    st.pyplot(fig7)
    plt.close()

    st.markdown("#### Correlation Heatmap")
    fig8, ax8 = plt.subplots(figsize=(7, 5))
    fig8.patch.set_facecolor('#0e1420')
    ax8.set_facecolor('#080c14')
    corr = df_raw[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                linewidths=0.5, ax=ax8, cbar_kws={'shrink': 0.8})
    ax8.set_title('Feature Correlation', color='white', fontweight='bold')
    ax8.tick_params(colors='#94a3b8')
    plt.tight_layout()
    st.pyplot(fig8)
    plt.close()


# ── TAB 4: DATA & EXPORT ─────────────────────
with tab4:
    st.markdown("#### Cluster Profile Summary")
    profile = df_out.groupby('Cluster')[selected_features].mean().round(2)
    profile['Count'] = df_out['Cluster'].value_counts().sort_index()
    profile['Segment'] = profile.index.map(labels_map)
    st.dataframe(profile, use_container_width=True)

    st.markdown("#### Full Labeled Dataset")
    st.dataframe(df_out, use_container_width=True, height=300)

    st.markdown("#### Export")
    csv_bytes = df_out.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇ Download Labeled Dataset (CSV)",
        data=csv_bytes,
        file_name="customer_segments_output.csv",
        mime="text/csv"
    )
