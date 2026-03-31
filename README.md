# 🎧 Amazon Music Clustering Dashboard

An interactive machine learning dashboard that analyzes and clusters music tracks based on audio features. Inspired by real-world music platforms like Amazon Music, this project helps uncover hidden patterns in songs using advanced clustering techniques.

---

## 🚀 Features

* 📊 Interactive dashboard built with Streamlit
* 🔍 Multiple clustering algorithms:

  * K-Means
  * DBSCAN
  * HDBSCAN (optional)
* 📉 PCA-based cluster visualization
* 🔥 Heatmap for feature comparison across clusters
* 🎯 Feature distribution analysis (boxplots)
* ⚡ Automatic numeric feature selection
* 🌙 Modern dark-themed UI

---

## 🧠 Project Overview

This project simulates how platforms like **Amazon Music** can group songs based on listening characteristics.

Using unsupervised learning, tracks are clustered based on features such as:

* Danceability
* Energy
* Loudness
* Speechiness
* Acousticness
* Instrumentalness
* Liveness
* Valence
* Tempo

The goal is to identify meaningful segments like:

* 🎉 Party songs
* 🌿 Chill songs
* 🏋️ Workout tracks
* 🎧 Experimental / Outliers

---

## 🛠️ Tech Stack

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* Seaborn & Matplotlib
* HDBSCAN (optional)

---

## 📂 Dataset

* Dataset used: `single_genre_artists.csv`
* Contains numerical audio features of tracks
* Preprocessing steps:

  * Selecting numeric columns
  * Handling missing values
  * Feature scaling using StandardScaler

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/amazon-music-clustering.git
cd amazon-music-clustering
```

### 2️⃣ Install dependencies

```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn hdbscan
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## 📊 How It Works

1. Load dataset
2. Extract numerical features
3. Scale features using StandardScaler
4. Apply clustering algorithm (KMeans / DBSCAN / HDBSCAN)
5. Reduce dimensions using PCA
6. Visualize clusters and analyze feature distributions

---

## 📈 Outputs

* Cluster visualization (PCA plot)
* Cluster distribution chart
* Feature-wise cluster comparison
* Heatmap of cluster averages
* Boxplot-based feature analysis

---

## ⚡ Insights

* DBSCAN & HDBSCAN detect **noise (outliers)**
* K-Means creates **clear segmented groups**
* PCA helps visualize complex multi-dimensional data
* Audio features strongly influence clustering behavior

---

## 🎯 Future Improvements

* Add 3D interactive visualizations (Plotly)
* Implement Silhouette Score / Elbow Method
* Deploy on Streamlit Cloud
* Add recommendation system

---

## 👨‍💻 Author

**nithish**
Aspiring Data Scientist

---

## ⭐ Acknowledgements

* Scikit-learn
* Streamlit
* Open music datasets

---

## 📌 Note

This project is for educational purposes and demonstrates how clustering can be applied in real-world music platforms like Amazon Music.

---
