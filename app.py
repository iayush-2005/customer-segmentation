import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Customer Segmentation · K-Means",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════════
# REDESIGNED STYLING
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@300;400;500;600&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    code, pre { font-family: 'JetBrains Mono', monospace !important; }
    
    #MainMenu, footer, header {visibility: hidden;}
    
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0f1524 50%, #0a0e1a 100%);
    }
    
    /* Hero */
    .hero-section {
        background: linear-gradient(135deg, rgba(0,212,255,0.05), rgba(124,58,237,0.05));
        border: 1px solid rgba(0,212,255,0.2);
        border-radius: 12px;
        padding: 48px 40px;
        margin-bottom: 32px;
        position: relative;
        overflow: hidden;
    }
    .hero-section::before {
        content: '';
        position: absolute;
        top: -2px; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, #00d4ff, #7c3aed, #00d4ff);
        animation: shimmer 3s infinite;
    }
    @keyframes shimmer { 0%, 100% { opacity: 0.5; } 50% { opacity: 1; } }
    
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d4ff, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 12px;
        line-height: 1.2;
    }
    
    .hero-subtitle {
        font-size: 1rem;
        color: #94a3b8;
        font-weight: 400;
    }
    
    .hero-badge {
        display: inline-block;
        padding: 6px 14px;
        background: rgba(0,212,255,0.1);
        border: 1px solid rgba(0,212,255,0.3);
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        color: #00d4ff;
        margin-right: 8px;
        margin-top: 16px;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Workflow Stepper */
    .workflow-container {
        background: rgba(15,21,36,0.6);
        border: 1px solid rgba(28,38,64,0.8);
        border-radius: 10px;
        padding: 32px;
        margin-bottom: 32px;
        backdrop-filter: blur(10px);
    }
    
    .workflow-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .workflow-icon {
        width: 32px; height: 32px;
        background: linear-gradient(135deg, #00d4ff, #7c3aed);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .workflow-steps {
        display: flex;
        gap: 16px;
        flex-wrap: wrap;
        margin-bottom: 20px;
    }
    
    .workflow-step {
        flex: 1;
        min-width: 160px;
        background: rgba(10,14,26,0.5);
        border: 1px solid rgba(28,38,64,0.6);
        border-radius: 8px;
        padding: 16px;
        position: relative;
        transition: all 0.3s;
    }
    
    .workflow-step.active {
        border-color: #00d4ff;
        background: rgba(0,212,255,0.05);
        box-shadow: 0 0 20px rgba(0,212,255,0.15);
    }
    
    .workflow-step.completed {
        border-color: #10b981;
        opacity: 0.7;
    }
    
    .step-number {
        position: absolute;
        top: -12px; left: 12px;
        width: 24px; height: 24px;
        background: #0a0e1a;
        border: 2px solid #00d4ff;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 11px;
        font-weight: 700;
        color: #00d4ff;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .workflow-step.completed .step-number {
        border-color: #10b981;
        color: #10b981;
    }
    
    .step-label {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
        font-weight: 600;
    }
    
    .step-title {
        font-size: 0.9rem;
        color: #e2e8f0;
        font-weight: 600;
    }
    
    .workflow-step.active .step-title { color: #00d4ff; }
    
    /* Action Cards */
    .action-card {
        background: linear-gradient(135deg, rgba(0,212,255,0.08), rgba(124,58,237,0.08));
        border: 1px solid rgba(0,212,255,0.3);
        border-radius: 10px;
        padding: 20px 24px;
        margin-bottom: 16px;
    }
    
    .action-card-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #00d4ff;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .action-card-text {
        font-size: 0.85rem;
        color: #94a3b8;
        line-height: 1.6;
    }
    
    /* Info Boxes */
    .info-box {
        background: rgba(77,159,255,0.08);
        border-left: 3px solid #4d9fff;
        padding: 14px 18px;
        border-radius: 6px;
        margin: 12px 0;
    }
    
    .info-box-title {
        font-size: 0.8rem;
        font-weight: 700;
        color: #4d9fff;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .info-box-text {
        font-size: 0.85rem;
        color: #cbd5e1;
        line-height: 1.6;
    }
    
    .success-box {
        background: rgba(16,185,129,0.08);
        border-left-color: #10b981;
    }
    .success-box .info-box-title { color: #10b981; }
    
    .warn-box {
        background: rgba(245,158,11,0.08);
        border-left-color: #f59e0b;
    }
    .warn-box .info-box-title { color: #f59e0b; }
    
    /* Metrics */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 12px;
        margin: 20px 0;
    }
    
    .metric-item {
        background: rgba(15,21,36,0.6);
        border: 1px solid rgba(28,38,64,0.8);
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d4ff, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    
    /* Loading Animation */
    .loading-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 60px 20px;
        gap: 20px;
    }
    
    .spinner {
        width: 60px; height: 60px;
        border: 3px solid rgba(0,212,255,0.1);
        border-top-color: #00d4ff;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin { to { transform: rotate(360deg); } }
    
    .loading-text {
        font-size: 0.9rem;
        color: #64748b;
        font-weight: 500;
    }
    
    .loading-substep {
        font-size: 0.8rem;
        color: #4a5580;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(15,21,36,0.4);
        padding: 8px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border: 1px solid transparent;
        color: #64748b;
        font-weight: 600;
        font-size: 0.85rem;
        padding: 10px 20px;
        border-radius: 6px;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(0,212,255,0.05);
        border-color: rgba(0,212,255,0.2);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0,212,255,0.15), rgba(124,58,237,0.15));
        border-color: #00d4ff;
        color: #00d4ff;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(10,14,26,0.95);
        border-right: 1px solid rgba(28,38,64,0.6);
    }
    
    section[data-testid="stSidebar"] > div { background: transparent; }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff, #7c3aed);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 10px 24px;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,212,255,0.3);
    }
    
    .stDownloadButton > button {
        background: rgba(16,185,129,0.15);
        border: 1px solid #10b981;
        color: #10b981;
        border-radius: 8px;
        font-weight: 600;
    }
    
    .stDownloadButton > button:hover {
        background: rgba(16,185,129,0.25);
    }
    
    h3 { color: #e2e8f0; font-weight: 700; margin-bottom: 16px; font-size: 1.2rem; }
    h4 { color: #cbd5e1; font-weight: 600; margin-bottom: 12px; font-size: 1rem; }
</style>
""", unsafe_allow_html=True)

PALETTE = ['#00d4ff', '#7c3aed', '#10b981', '#f59e0b', '#ef4444',
           '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#a855f7']

# ═══════════════════════════════════════════════════════════════════════════
# FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_uploaded_dataset(file_bytes, file_name):
    df = pd.read_csv(io.BytesIO(file_bytes))
    df.columns = df.columns.str.strip().str.replace(' ', '_')
    return df


def run_pipeline(df, feature_cols, k, progress_container=None):
    if progress_container:
        with progress_container:
            st.markdown('<div class="loading-container"><div class="spinner"></div><div class="loading-text">Running K-Means Pipeline...</div>', unsafe_allow_html=True)
            step_text = st.empty()
            st.markdown('</div>', unsafe_allow_html=True)
    
    def update_step(msg):
        if progress_container:
            step_text.markdown(f'<div class="loading-substep">{msg}</div>', unsafe_allow_html=True)
            time.sleep(0.4)
    
    update_step("Step 1/6: Extracting features...")
    X = df[list(feature_cols)].values
    
    update_step("Step 2/6: Handling missing values...")
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    
    update_step("Step 3/6: Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    update_step("Step 4/6: Computing elbow curve (K=2 to 10)...")
    wcss, sil = [], []
    for ki in range(2, 11):
        km = KMeans(n_clusters=ki, init='k-means++', n_init=10, random_state=42)
        lbl = km.fit_predict(X_scaled)
        wcss.append(km.inertia_)
        sil.append(silhouette_score(X_scaled, lbl))
    
    update_step(f"Step 5/6: Training final model (K={k})...")
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    sil_final = silhouette_score(X_scaled, labels)
    
    update_step("Step 6/6: Generating cluster labels...")
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
    
    if len(feature_cols) > 2:
        pca = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(X_scaled)
        ev = pca.explained_variance_ratio_.sum() * 100
    else:
        X_2d = X_scaled
        ev = 100.0
    
    return {
        'df': result_df, 'X_scaled': X_scaled, 'X_2d': X_2d, 'ev': ev,
        'wcss': wcss, 'sil': sil, 'sil_final': sil_final,
        'centers_df': centers_df, 'labels_map': labels_map,
        'inertia': kmeans.inertia_, 'feature_cols': list(feature_cols),
    }


def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#0a0e1a')
    buf.seek(0)
    return buf.read()


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🎯 Configuration Panel")
    st.markdown("---")
    
    st.markdown("**Step 1: Upload Dataset**")
    st.caption("CSV with ≥2 numeric columns")
    
    uploaded = st.file_uploader("Choose CSV", type=["csv"], label_visibility="collapsed")
    
    df_raw = None
    if uploaded:
        try:
            file_bytes = uploaded.read()
            df_raw = load_uploaded_dataset(file_bytes, uploaded.name)
            st.success(f"✓ {uploaded.name}")
            st.caption(f"**{df_raw.shape[0]}** rows · **{df_raw.shape[1]}** cols")
        except Exception as e:
            st.error(f"❌ {e}")
    else:
        st.info("📂 No file uploaded")
        with st.expander("💡 Need sample data?"):
            st.markdown("[Mall Customers Dataset →](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)")
    
    if df_raw is not None:
        st.markdown("---")
        st.markdown("**Step 2: Select Features**")
        numeric_cols = df_raw.select_dtypes(include=np.number).columns.tolist()
        id_like = [c for c in numeric_cols if any(x in c.lower() for x in ['id', 'index', 'no'])]
        default_feats = [c for c in numeric_cols if c not in id_like][-2:]
        
        selected_features = st.multiselect(
            "Numeric columns",
            options=[c for c in numeric_cols if c not in id_like],
            default=default_feats,
            help="Select 2+ features"
        )
        
        st.markdown("---")
        st.markdown("**Step 3: Choose K**")
        k_value = st.slider("Clusters", 2, 10, 5, help="Number of segments")
        
        st.markdown("---")
        st.caption("**18B1WCI675 · DMDW Lab**")
        st.caption("JUIT Waknaghat")

# ═══════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-section">
    <div class="hero-title">🎯 Customer Segmentation Engine</div>
    <div class="hero-subtitle">AI-powered clustering · Discover hidden patterns in customer behavior</div>
    <span class="hero-badge">K-Means</span>
    <span class="hero-badge">Unsupervised ML</span>
    <span class="hero-badge">DMDW Lab</span>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# STATE 1: No Data
# ═══════════════════════════════════════════════════════════════════════════
if df_raw is None:
    st.markdown("""
    <div class="workflow-container">
        <div class="workflow-header">
            <div class="workflow-icon">📋</div>
            <span>Workflow</span>
        </div>
        <div class="workflow-steps">
            <div class="workflow-step active">
                <div class="step-number">1</div>
                <div class="step-label">Upload</div>
                <div class="step-title">Load CSV</div>
            </div>
            <div class="workflow-step">
                <div class="step-number">2</div>
                <div class="step-label">Select</div>
                <div class="step-title">Pick Features</div>
            </div>
            <div class="workflow-step">
                <div class="step-number">3</div>
                <div class="step-label">Configure</div>
                <div class="step-title">Set K Value</div>
            </div>
            <div class="workflow-step">
                <div class="step-number">4</div>
                <div class="step-label">Analyze</div>
                <div class="step-title">View Results</div>
            </div>
        </div>
    </div>
    
    <div class="action-card">
        <div class="action-card-title">👈 What to do first?</div>
        <div class="action-card-text">
        Upload a CSV file using the sidebar. Your dataset should have at least 2 numeric columns 
        representing customer attributes (e.g., Income, Spending Score, Age).
        </div>
    </div>
    
    <div class="info-box">
        <div class="info-box-title">💡 How K-Means Works</div>
        <div class="info-box-text">
        K-Means groups similar customers into K clusters by iteratively assigning data points 
        to the nearest centroid and recalculating centroids until convergence. This reveals 
        natural customer segments for targeted marketing.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="info-box success-box">
            <div class="info-box-title">✓ What You Get</div>
            <div class="info-box-text">
            • Interactive cluster visualizations<br/>
            • Elbow curve for optimal K<br/>
            • Silhouette score analysis<br/>
            • Downloadable labeled CSV<br/>
            • Business insights per segment
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box warn-box">
            <div class="info-box-title">⚠ Requirements</div>
            <div class="info-box-text">
            • CSV format<br/>
            • ≥2 numeric columns<br/>
            • ≥50 rows recommended<br/>
            • Missing values auto-imputed
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════
# STATE 2: Data Loaded, Features Not Selected
# ═══════════════════════════════════════════════════════════════════════════
elif not selected_features or len(selected_features) < 2:
    st.markdown("""
    <div class="workflow-container">
        <div class="workflow-header">
            <div class="workflow-icon">📋</div>
            <span>Workflow</span>
        </div>
        <div class="workflow-steps">
            <div class="workflow-step completed">
                <div class="step-number">✓</div>
                <div class="step-label">Upload</div>
                <div class="step-title">Load CSV</div>
            </div>
            <div class="workflow-step active">
                <div class="step-number">2</div>
                <div class="step-label">Select</div>
                <div class="step-title">Pick Features</div>
            </div>
            <div class="workflow-step">
                <div class="step-number">3</div>
                <div class="step-label">Configure</div>
                <div class="step-title">Set K Value</div>
            </div>
            <div class="workflow-step">
                <div class="step-number">4</div>
                <div class="step-label">Analyze</div>
                <div class="step-title">View Results</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="action-card">
        <div class="action-card-title">✅ Dataset Loaded: {df_raw.shape[0]} × {df_raw.shape[1]}</div>
        <div class="action-card-text">
        Great! Now select at least 2 numeric features from the sidebar. These should represent 
        meaningful customer attributes that help distinguish different groups.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📊 Dataset Preview")
    st.dataframe(df_raw.head(10), use_container_width=True, height=300)
    
    st.markdown("### 🔢 Numeric Columns Available")
    numeric_cols = df_raw.select_dtypes(include=np.number).columns.tolist()
    col_data = []
    for col in numeric_cols:
        col_data.append({
            'Column': col,
            'Mean': f"{df_raw[col].mean():.2f}",
            'Std': f"{df_raw[col].std():.2f}",
            'Min': f"{df_raw[col].min():.2f}",
            'Max': f"{df_raw[col].max():.2f}",
            'Nulls': int(df_raw[col].isnull().sum())
        })
    st.dataframe(pd.DataFrame(col_data), use_container_width=True, hide_index=True)
    
    st.markdown("""
    <div class="info-box">
        <div class="info-box-title">💡 Feature Selection Tips</div>
        <div class="info-box-text">
        • Choose features with different scales (e.g., Income + Spending)<br/>
        • Avoid ID columns<br/>
        • Features are auto-scaled to prevent bias
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════
# STATE 3: Ready to Run
# ═══════════════════════════════════════════════════════════════════════════
else:
    st.markdown("""
    <div class="workflow-container">
        <div class="workflow-header">
            <div class="workflow-icon">📋</div>
            <span>Workflow</span>
        </div>
        <div class="workflow-steps">
            <div class="workflow-step completed">
                <div class="step-number">✓</div>
                <div class="step-label">Upload</div>
                <div class="step-title">Load CSV</div>
            </div>
            <div class="workflow-step completed">
                <div class="step-number">✓</div>
                <div class="step-label">Select</div>
                <div class="step-title">Pick Features</div>
            </div>
            <div class="workflow-step completed">
                <div class="step-number">✓</div>
                <div class="step-label">Configure</div>
                <div class="step-title">Set K Value</div>
            </div>
            <div class="workflow-step active">
                <div class="step-number">4</div>
                <div class="step-label">Analyze</div>
                <div class="step-title">View Results</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Run pipeline with loading
progress_placeholder = st.empty()
results = run_pipeline(df_raw, tuple(selected_features), k_value, progress_placeholder)
progress_placeholder.empty()

df_out = results['df']
labels_map = results['labels_map']

# SUCCESS
st.markdown(f"""
<div class="action-card" style="background: linear-gradient(135deg, rgba(16,185,129,0.12), rgba(0,212,255,0.12)); border-color: rgba(16,185,129,0.4);">
    <div class="action-card-title" style="color: #10b981;">✅ Clustering Complete!</div>
    <div class="action-card-text">
    K-Means identified <strong>{k_value}</strong> customer segments. 
    Explore visualizations, download results, and derive business insights below.
    </div>
</div>
""", unsafe_allow_html=True)

# METRICS
st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
cols = st.columns(4)
with cols[0]:
    st.markdown(f'<div class="metric-item"><div class="metric-value">{k_value}</div><div class="metric-label">Clusters</div></div>', unsafe_allow_html=True)
with cols[1]:
    st.markdown(f'<div class="metric-item"><div class="metric-value">{results["sil_final"]:.3f}</div><div class="metric-label">Silhouette</div></div>', unsafe_allow_html=True)
with cols[2]:
    st.markdown(f'<div class="metric-item"><div class="metric-value">{int(results["inertia"])}</div><div class="metric-label">WCSS</div></div>', unsafe_allow_html=True)
with cols[3]:
    st.markdown(f'<div class="metric-item"><div class="metric-value">{len(df_out)}</div><div class="metric-label">Customers</div></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# TABS
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Cluster Visualizations",
    "📉 Elbow & Silhouette",
    "🔍 EDA",
    "📋 Data Export"
])

# TAB 1: Cluster Visualizations
with tab1:
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("#### Cluster Scatter")
        fig, ax = plt.subplots(figsize=(8, 5.5))
        fig.patch.set_facecolor('#0a0e1a')
        ax.set_facecolor('#0f1524')
        
        f1, f2 = selected_features[0], selected_features[-1]
        for cid in sorted(df_out['Cluster'].unique()):
            mask = df_out['Cluster'] == cid
            ax.scatter(df_out.loc[mask, f1], df_out.loc[mask, f2],
                      s=65, alpha=0.85, color=PALETTE[cid % len(PALETTE)],
                      label=f'C{cid}: {labels_map[cid][:28]}')
        
        cx, cy = results['centers_df'][f1], results['centers_df'][f2]
        ax.scatter(cx, cy, s=280, marker='X', c='white',
                  edgecolors='black', linewidths=1.5, zorder=5, label='Centroids')
        
        ax.set_xlabel(f1.replace('_', ' '), color='#94a3b8')
        ax.set_ylabel(f2.replace('_', ' '), color='#94a3b8')
        ax.tick_params(colors='#64748b')
        ax.spines[:].set_color('#1e2d4a')
        ax.legend(fontsize=7.5, facecolor='#0e1420', edgecolor='#1e2d4a', labelcolor='white')
        ax.grid(alpha=0.15, color='#1e2d4a')
        ax.set_title(f'K-Means Clusters (K={k_value})', color='white', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        st.download_button("⬇ Download", fig_to_bytes(fig), "scatter.png", "image/png")
        plt.close()
    
    with col2:
        st.markdown("#### Cluster Sizes")
        fig2, ax2 = plt.subplots(figsize=(5, 5.5))
        fig2.patch.set_facecolor('#0a0e1a')
        ax2.set_facecolor('#0f1524')
        
        counts = df_out['Cluster'].value_counts().sort_index()
        bars = ax2.bar([f'C{i}' for i in counts.index], counts.values,
                      color=[PALETTE[i % len(PALETTE)] for i in counts.index],
                      edgecolor='white', alpha=0.9, width=0.6)
        for bar, val in zip(bars, counts.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(val), ha='center', va='bottom', color='white', fontweight='bold')
        
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
        fig3.patch.set_facecolor('#0a0e1a')
        ax3.set_facecolor('#0f1524')
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
        ax3.set_title('PCA Projection', color='white', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()
    
    # Boxplots
    st.markdown("#### Feature Distribution by Cluster")
    ncols = min(len(selected_features), 3)
    fig4, axes4 = plt.subplots(1, ncols, figsize=(5 * ncols, 4))
    fig4.patch.set_facecolor('#0a0e1a')
    if ncols == 1:
        axes4 = [axes4]
    for ax4, feat in zip(axes4, selected_features[:ncols]):
        ax4.set_facecolor('#0f1524')
        data_list = [df_out[df_out['Cluster'] == c][feat].values
                     for c in sorted(df_out['Cluster'].unique())]
        bp = ax4.boxplot(data_list, patch_artist=True)
        for patch, color in zip(bp['boxes'], PALETTE):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color='white')
        ax4.set_title(feat.replace('_', ' '), color='white', fontweight='bold')
        ax4.set_xticklabels([f'C{i}' for i in sorted(df_out['Cluster'].unique())],
                            color='#64748b')
        ax4.tick_params(colors='#64748b')
        ax4.spines[:].set_color('#1e2d4a')
        ax4.grid(axis='y', alpha=0.15, color='#1e2d4a')
    plt.tight_layout()
    st.pyplot(fig4)
    st.download_button("⬇ Download Boxplots", fig_to_bytes(fig4), "boxplots.png", "image/png")
    plt.close()
    
    st.markdown("""
    <div class="info-box">
        <div class="info-box-title">📊 Interpretation Guide</div>
        <div class="info-box-text">
        <strong>Scatter Plot:</strong> Shows customer distribution. Tighter clusters = stronger separation.<br/>
        <strong>Centroids (X markers):</strong> Average customer in each segment — use for targeting.<br/>
        <strong>Boxplots:</strong> Compare feature ranges across segments. Wide boxes = diverse customers within cluster.
        </div>
    </div>
    """, unsafe_allow_html=True)

# TAB 2: Elbow
with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Elbow Curve")
        fig, ax = plt.subplots(figsize=(6.5, 4))
        fig.patch.set_facecolor('#0a0e1a')
        ax.set_facecolor('#0f1524')
        k_range = list(range(2, 11))
        ax.plot(k_range, results['wcss'], 'o-', color='#00d4ff', linewidth=2.5, markersize=8)
        ax.fill_between(k_range, results['wcss'], alpha=0.08, color='#00d4ff')
        ax.axvline(x=k_value, color='#ef4444', linestyle='--', alpha=0.8, label=f'Selected K={k_value}')
        ax.set_xlabel('K', color='#94a3b8')
        ax.set_ylabel('WCSS', color='#94a3b8')
        ax.tick_params(colors='#64748b')
        ax.spines[:].set_color('#1e2d4a')
        ax.grid(alpha=0.15, color='#1e2d4a')
        ax.legend(facecolor='#0e1420', edgecolor='#1e2d4a', labelcolor='white')
        ax.set_title('Elbow Method', color='white', fontweight='bold')
        ax.set_xticks(k_range)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.markdown("#### Silhouette Score")
        fig, ax = plt.subplots(figsize=(6.5, 4))
        fig.patch.set_facecolor('#0a0e1a')
        ax.set_facecolor('#0f1524')
        best_k = k_range[results['sil'].index(max(results['sil']))]
        ax.plot(k_range, results['sil'], 's-', color='#10b981', linewidth=2.5, markersize=8)
        ax.fill_between(k_range, results['sil'], alpha=0.08, color='#10b981')
        ax.axvline(x=best_k, color='#f59e0b', linestyle='--', alpha=0.8, label=f'Best K={best_k}')
        ax.axvline(x=k_value, color='#ef4444', linestyle='--', alpha=0.8, label=f'Selected K={k_value}')
        ax.set_xlabel('K', color='#94a3b8')
        ax.set_ylabel('Silhouette', color='#94a3b8')
        ax.tick_params(colors='#64748b')
        ax.spines[:].set_color('#1e2d4a')
        ax.grid(alpha=0.15, color='#1e2d4a')
        ax.legend(facecolor='#0e1420', edgecolor='#1e2d4a', labelcolor='white')
        ax.set_title('Silhouette', color='white', fontweight='bold')
        ax.set_xticks(k_range)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    st.markdown("#### Scores Table")
    score_df = pd.DataFrame({
        'K': k_range,
        'WCSS': [f'{w:.2f}' for w in results['wcss']],
        'Silhouette': [f'{s:.4f}' for s in results['sil']]
    })
    best_k = k_range[results['sil'].index(max(results['sil']))]
    score_df['Note'] = score_df['K'].apply(
        lambda x: '⭐ Best Silhouette' if x == best_k else
                  ('✅ Selected' if x == k_value else ''))
    st.dataframe(score_df, use_container_width=True, hide_index=True)
    
    st.markdown("""
    <div class="info-box">
        <div class="info-box-title">📉 How to Choose Optimal K</div>
        <div class="info-box-text">
        <strong>Elbow Method:</strong> Look for the "elbow" — where WCSS drops sharply then flattens. That K balances fit vs. complexity.<br/>
        <strong>Silhouette Score:</strong> Measures cluster separation. Range: -1 to 1. Higher = better separation. Aim for >0.5.<br/>
        <strong>Best Practice:</strong> If elbow is unclear, use the K with highest silhouette score.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box warn-box">
        <div class="info-box-title">⚠ Your Results</div>
        <div class="info-box-text">
        <strong>Selected K = {}</strong> | Silhouette Score = <strong>{:.4f}</strong><br/>
        {}
        </div>
    </div>
    """.format(
        k_value, 
        results['sil_final'],
        "Good separation — clusters are well-defined." if results['sil_final'] > 0.5 else
        "Moderate separation — consider trying K={} for better results.".format(best_k)
    ), unsafe_allow_html=True)

# TAB 3: EDA
with tab3:
    st.markdown("#### Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df_raw.shape[0])
    c2.metric("Columns", df_raw.shape[1])
    c3.metric("Nulls", int(df_raw.isnull().sum().sum()))
    st.dataframe(df_raw.describe().round(2), use_container_width=True)
    
    st.markdown("#### Feature Distributions")
    num_cols = df_raw.select_dtypes(include=np.number).columns.tolist()
    ncols2 = min(len(num_cols), 3)
    fig7, axes7 = plt.subplots(1, ncols2, figsize=(5.5 * ncols2, 4))
    fig7.patch.set_facecolor('#0a0e1a')
    if ncols2 == 1:
        axes7 = [axes7]
    for ax7, col, color in zip(axes7, num_cols[:ncols2], PALETTE):
        ax7.set_facecolor('#0f1524')
        ax7.hist(df_raw[col].dropna(), bins=20, color=color,
                edgecolor='white', alpha=0.85)
        ax7.set_title(col.replace('_', ' '), color='white', fontweight='bold')
        ax7.tick_params(colors='#64748b')
        ax7.spines[:].set_color('#1e2d4a')
        ax7.grid(axis='y', alpha=0.15, color='#1e2d4a')
    plt.tight_layout()
    st.pyplot(fig7)
    plt.close()
    
    st.markdown("#### Correlation Heatmap")
    fig8, ax8 = plt.subplots(figsize=(7, 5))
    fig8.patch.set_facecolor('#0a0e1a')
    ax8.set_facecolor('#0f1524')
    corr = df_raw[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                linewidths=0.5, ax=ax8, cbar_kws={'shrink': 0.8})
    ax8.set_title('Feature Correlation', color='white', fontweight='bold')
    ax8.tick_params(colors='#94a3b8')
    plt.tight_layout()
    st.pyplot(fig8)
    plt.close()
    
    st.markdown("""
    <div class="info-box">
        <div class="info-box-title">📊 Data Quality Checks</div>
        <div class="info-box-text">
        <strong>Distributions:</strong> Look for outliers (extreme values) or skewed data. K-Means assumes roughly spherical clusters.<br/>
        <strong>Correlation:</strong> High correlation (>0.7) means features are redundant. Consider removing one.<br/>
        <strong>Missing Values:</strong> Auto-imputed with mean — verify this makes sense for your domain.
        </div>
    </div>
    """, unsafe_allow_html=True)

# TAB 4: Export
with tab4:
    st.markdown("#### Cluster Profile")
    profile = df_out.groupby('Cluster')[selected_features].mean().round(2)
    profile['Count'] = df_out['Cluster'].value_counts().sort_index()
    profile['Segment'] = profile.index.map(labels_map)
    st.dataframe(profile, use_container_width=True)
    
    st.markdown("#### Business Insights")
    st.markdown("""
    <div class="info-box success-box">
        <div class="info-box-title">💡 Marketing Recommendations</div>
        <div class="info-box-text">
        Use cluster centroids to define customer personas. Each segment represents a unique customer type 
        with distinct characteristics. Tailor marketing campaigns, product offerings, and pricing strategies per segment.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate segment insights
    for cluster_id in sorted(df_out['Cluster'].unique()):
        grp = df_out[df_out['Cluster'] == cluster_id]
        size = len(grp)
        pct = size / len(df_out) * 100
        
        with st.expander(f"🎯 Cluster {cluster_id}: {labels_map[cluster_id]} ({size} customers, {pct:.1f}%)"):
            cols_insight = st.columns([2, 1])
            with cols_insight[0]:
                st.markdown("**Profile Summary:**")
                for feat in selected_features:
                    avg_val = grp[feat].mean()
                    st.caption(f"• Average {feat.replace('_', ' ')}: **{avg_val:.2f}**")
            with cols_insight[1]:
                st.markdown("**Recommended Action:**")
                # Simple heuristic for recommendations
                f1_avg = grp[selected_features[0]].mean()
                f2_avg = grp[selected_features[-1]].mean()
                f1_median = df_out[selected_features[0]].median()
                f2_median = df_out[selected_features[-1]].median()
                
                if f1_avg > f1_median and f2_avg > f2_median:
                    st.info("🌟 **Premium Segment** — High value targets for upselling and loyalty programs")
                elif f1_avg < f1_median and f2_avg < f2_median:
                    st.warning("💰 **Budget Segment** — Focus on volume sales and value propositions")
                elif f1_avg > f1_median and f2_avg < f2_median:
                    st.success("🎯 **Growth Potential** — Engage with targeted offers to increase spending")
                else:
                    st.info("📊 **Mid-Tier Segment** — Maintain engagement with balanced strategies")
    
    st.markdown("#### Full Labeled Dataset")
    st.dataframe(df_out, use_container_width=True, height=300)
    
    st.markdown("#### Download Options")
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        csv_bytes = df_out.to_csv(index=False).encode('utf-8')
        st.download_button("⬇ Download Full Dataset (CSV)", csv_bytes, "segmented_customers.csv", "text/csv", use_container_width=True)
    with col_dl2:
        profile_csv = profile.to_csv().encode('utf-8')
        st.download_button("⬇ Download Cluster Summary (CSV)", profile_csv, "cluster_summary.csv", "text/csv", use_container_width=True)
    
    st.markdown("""
    <div class="info-box warn-box">
        <div class="info-box-title">🔄 Next Steps</div>
        <div class="info-box-text">
        1. <strong>Validate segments</strong> with domain experts to ensure they make business sense<br/>
        2. <strong>A/B test</strong> targeted campaigns on different segments to measure lift<br/>
        3. <strong>Monitor drift</strong> — re-cluster quarterly to catch changing customer behavior<br/>
        4. <strong>Enrich data</strong> — add more features (purchase history, demographics) for deeper segmentation
        </div>
    </div>
    """, unsafe_allow_html=True)
