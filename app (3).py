import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN

# Try HDBSCAN safely
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except:
    HDBSCAN_AVAILABLE = False

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Music Clustering Dashboard", layout="wide")

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>
.stApp {
    background-color: #0E1117;
    color: white;
}
.card {
    background-color: #1E1E1E;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
}
.card h3 { color: #aaa; font-size: 14px; }
.card h1 { color: #00C853; font-size: 30px; }
.section-title {
    font-size: 24px;
    font-weight: bold;
    margin-top: 25px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.markdown("<h1 style='text-align:center;'>🎧 Music Clustering Dashboard</h1>", unsafe_allow_html=True)

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("single_genre_artists.csv")

df = load_data()

# -----------------------------
# CLEAN DATA
# -----------------------------
df_numeric = df.select_dtypes(include=np.number).dropna()

# -----------------------------
# FEATURES
# -----------------------------
features = df_numeric.columns.tolist()
X = df_numeric

# -----------------------------
# SCALING
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("⚙️ Settings")

algorithm = st.sidebar.selectbox(
    "Select Algorithm",
    ["KMeans", "DBSCAN"] + (["HDBSCAN"] if HDBSCAN_AVAILABLE else [])
)

# Parameters
if algorithm == "KMeans":
    k = st.sidebar.slider("Clusters", 2, 6, 3)

elif algorithm == "DBSCAN":
    eps = st.sidebar.slider("EPS", 0.5, 5.0, 1.5)
    min_samples = st.sidebar.slider("Min Samples", 5, 50, 10)

elif algorithm == "HDBSCAN":
    min_cluster_size = st.sidebar.slider("Min Cluster Size", 5, 50, 10)

# -----------------------------
# MODEL
# -----------------------------
if algorithm == "KMeans":
    model = KMeans(n_clusters=k, random_state=42)
    df['Cluster'] = model.fit_predict(X_scaled)

elif algorithm == "DBSCAN":
    model = DBSCAN(eps=eps, min_samples=min_samples)
    df['Cluster'] = model.fit_predict(X_scaled)

elif algorithm == "HDBSCAN":
    model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    df['Cluster'] = model.fit_predict(X_scaled)

# -----------------------------
# METRICS
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"<div class='card'><h3>Total Rows</h3><h1>{len(df)}</h1></div>", unsafe_allow_html=True)

with col2:
    st.markdown(f"<div class='card'><h3>Features</h3><h1>{len(features)}</h1></div>", unsafe_allow_html=True)

with col3:
    st.markdown(f"<div class='card'><h3>Algorithm</h3><h1>{algorithm}</h1></div>", unsafe_allow_html=True)

with col4:
    st.markdown(f"<div class='card'><h3>Clusters</h3><h1>{df['Cluster'].nunique()}</h1></div>", unsafe_allow_html=True)

st.markdown("---")

# -----------------------------
# PCA
# -----------------------------
st.markdown("<div class='section-title'>📊 Cluster Visualization</div>", unsafe_allow_html=True)

pca = PCA(n_components=2)
pca_data = pca.fit_transform(X_scaled)

fig, ax = plt.subplots()
fig.patch.set_facecolor('#0E1117')
ax.set_facecolor('#0E1117')

scatter = ax.scatter(
    pca_data[:, 0],
    pca_data[:, 1],
    c=df['Cluster'],
    cmap='viridis'
)

ax.set_xlabel("PCA 1", color='white')
ax.set_ylabel("PCA 2", color='white')
ax.tick_params(colors='white')

st.pyplot(fig)

# -----------------------------
# DISTRIBUTION
# -----------------------------
st.markdown("<div class='section-title'>📊 Cluster Distribution</div>", unsafe_allow_html=True)

fig2, ax2 = plt.subplots()
fig2.patch.set_facecolor('#0E1117')
ax2.set_facecolor('#0E1117')

sns.countplot(x='Cluster', data=df, ax=ax2)
ax2.tick_params(colors='white')

st.pyplot(fig2)

# -----------------------------
# MEANS
# -----------------------------
st.markdown("<div class='section-title'>📋 Cluster Feature Means</div>", unsafe_allow_html=True)

cluster_means = df.groupby('Cluster')[features].mean()
st.dataframe(cluster_means)

# -----------------------------
# HEATMAP
# -----------------------------
st.markdown("<div class='section-title'>🔥 Heatmap</div>", unsafe_allow_html=True)

fig3, ax3 = plt.subplots(figsize=(10,5))
sns.heatmap(cluster_means, annot=True, cmap="coolwarm", ax=ax3)

fig3.patch.set_facecolor('#0E1117')
ax3.set_facecolor('#0E1117')

st.pyplot(fig3)

# -----------------------------
# FEATURE ANALYSIS
# -----------------------------
st.markdown("<div class='section-title'>🎯 Feature Analysis</div>", unsafe_allow_html=True)

selected_feature = st.selectbox("Select Feature", features)

fig4, ax4 = plt.subplots()
fig4.patch.set_facecolor('#0E1117')
ax4.set_facecolor('#0E1117')

sns.boxplot(x='Cluster', y=selected_feature, data=df, ax=ax4)
ax4.tick_params(colors='white')

st.pyplot(fig4)

# -----------------------------
# INSIGHTS
# -----------------------------
noise_count = sum(df['Cluster'] == -1)

st.markdown(f"""
### ⚡ Insights
- 🔴 Noise Points: {noise_count}
- 🔵 Largest Cluster: {df['Cluster'].value_counts().idxmax()}
""")

# -----------------------------
# SAMPLE DATA
# -----------------------------
st.markdown("<div class='section-title'>📄 Sample Data</div>", unsafe_allow_html=True)

st.dataframe(df.head(20))