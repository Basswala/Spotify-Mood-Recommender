# 🎧 Spotify Mood Recommender

A machine learning project that clusters songs by **mood** using Spotify audio features.  
The goal is to recommend tracks with a similar vibe, based on features like energy, valence, danceability, and tempo.  
Built with a clean, reproducible structure — ready for portfolio presentation.

---

## 📌 Overview
- **Problem:** Traditional playlists rely on genre/artist. Here, we group songs by **mood**.  
- **Solution:** Use unsupervised learning (KMeans) on Spotify's audio features → assign mood clusters.  
- **Outcome:** A Streamlit app where a user inputs a track and gets mood-based recommendations.

---

## 📊 Dataset
- **Source:** Spotify API + provided dataset of audio features.  
- **Features:**  
  `danceability, energy, valence, tempo, loudness, speechiness, acousticness, instrumentalness, liveness, mode, key`

---

## 🧭 Workflow
1. **Data ingestion** → fetch/load Spotify audio features.  
2. **Feature engineering** → clean, scale, encode circular features.  
3. **Clustering** → KMeans (later DBSCAN/HDBSCAN) for mood groups.  
4. **Recommendation engine** → nearest neighbors within clusters.  
5. **Visualization** → radar charts, PCA/UMAP scatter plots.  
6. **Deployment** → Streamlit app for interactive use.

---

## ⚙️ Tech Stack
- **Python** (3.11+)  
- **Libraries:** `pandas`, `scikit-learn`, `spotipy`, `matplotlib`, `plotly`, `streamlit`, `umap-learn`, `joblib`  
- **Environment:** Managed with `uv`  

---

## 📂 Project Structure
```text
spotify-mood/
├── data/                 # datasets (raw + cached audio features)
├── src/
│   ├── io_spotify.py     # Spotify API auth + fetch
│   ├── features.py       # cleaning + scaling
│   ├── cluster.py        # KMeans clustering + labeling
│   ├── nn.py             # recommender (nearest neighbors)
│   ├── viz.py            # radar + PCA scatter plots
│   └── app/ui.py         # Streamlit app
├── models/               # scaler.pkl, kmeans.pkl, nn.index
├── notebooks/            # exploratory analysis
├── tests/                # unit tests
├── requirements.txt      # dependencies
├── .gitignore            # ignore rules
├── README.md             # this file
└── LICENSE
```

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Install dependencies
uv add pandas scikit-learn spotipy matplotlib plotly streamlit umap-learn joblib tqdm

# Or install from requirements.txt
pip install -r requirements.txt
```

### 2. Run the Pipeline
```bash
# Run the complete pipeline
python main.py
```

### 3. Launch the Web App
```bash
# Start the Streamlit application
streamlit run src/app/ui.py
```

### 4. Explore the Data
```bash
# Run Jupyter notebooks for analysis
jupyter notebook notebooks/
```

---

## 🔧 Usage

### Command Line Interface
```bash
# Run complete pipeline
python main.py

# Run specific components
python -c "from src.features import AudioFeatureProcessor; print('Features module loaded')"
```

### Programmatic Usage
```python
from src.features import AudioFeatureProcessor
from src.cluster import MoodClusterer
from src.nn import MoodRecommender

# Load and process data
df = pd.read_csv('data/dataset.csv')
processor = AudioFeatureProcessor()
df_processed = processor.fit_transform(df)

# Cluster songs by mood
clusterer = MoodClusterer(algorithm='kmeans')
clusterer.fit(df_processed[feature_cols].values, auto_tune=True)

# Build recommendation system
recommender = MoodRecommender()
recommender.fit(X, df_processed, clusterer.cluster_labels)

# Get recommendations
recommendations = recommender.recommend_similar_songs('track_id', n_recommendations=10)
```

### Web Interface
1. Launch the Streamlit app: `streamlit run src/app/ui.py`
2. Search for a track by name, artist, or album
3. View track analysis with radar charts
4. Get mood-based recommendations
5. Explore cluster visualizations

---

## 📈 Features

### 🎭 Mood Clustering
- **KMeans clustering** with automatic parameter tuning
- **Mood labeling** based on audio feature patterns
- **Cluster analysis** with statistics and visualizations

### 🎵 Recommendation System
- **Nearest neighbors** within mood clusters
- **Multiple distance metrics** (euclidean, cosine, manhattan)
- **Similarity scoring** for recommendation ranking

### 📊 Visualizations
- **Radar charts** for audio feature comparison
- **PCA/t-SNE/UMAP** scatter plots for cluster visualization
- **Feature distribution** plots by mood cluster
- **Cluster heatmaps** showing average features

### 🌐 Web Application
- **Interactive search** for tracks
- **Real-time recommendations** with similarity scores
- **Cluster exploration** with visualizations
- **Track analysis** with audio feature breakdowns

---

## 🧪 Testing

### Run Unit Tests
```bash
# Run all tests
pytest tests/

# Run specific test modules
pytest tests/test_features.py
pytest tests/test_cluster.py
pytest tests/test_nn.py

# Run with coverage
pytest --cov=src tests/
```

### Test Coverage
- **Feature processing** (outlier detection, scaling, encoding)
- **Clustering algorithms** (KMeans, parameter tuning, evaluation)
- **Recommendation system** (nearest neighbors, similarity scoring)
- **Visualization components** (charts, plots, interactive elements)

---

## 📊 Model Performance

### Clustering Quality
- **Silhouette Score**: Measures cluster separation
- **Calinski-Harabasz Index**: Cluster compactness
- **Davies-Bouldin Index**: Cluster validity

### Recommendation Accuracy
- **Similarity scoring** based on audio feature distance
- **Mood consistency** within cluster recommendations
- **Diversity metrics** for recommendation variety

---

## 🔍 Exploratory Analysis

### Notebooks
- **01_data_exploration.ipynb**: Dataset analysis and feature distributions
- **02_clustering_analysis.ipynb**: Clustering implementation and evaluation

### Key Insights
- **Feature correlations**: Energy-valence relationship, tempo-danceability
- **Genre patterns**: Audio feature distributions across genres
- **Mood clusters**: Distinct audio feature patterns for different moods

---

## 🛠️ Development

### Code Structure
- **Modular design** with separate concerns
- **Type hints** and comprehensive docstrings
- **Error handling** with informative messages
- **Logging** for debugging and monitoring

### Best Practices
- **Reproducible results** with random seeds
- **Model persistence** with joblib
- **Data validation** and preprocessing
- **Performance optimization** with efficient algorithms

---

## 📝 API Reference

### AudioFeatureProcessor
```python
processor = AudioFeatureProcessor(scaler_type='standard')
df_processed = processor.fit_transform(df)
```

### MoodClusterer
```python
clusterer = MoodClusterer(algorithm='kmeans')
clusterer.fit(X, auto_tune=True)
```

### MoodRecommender
```python
recommender = MoodRecommender(algorithm='ball_tree', metric='cosine')
recommender.fit(X, track_data, cluster_labels)
recommendations = recommender.recommend_similar_songs(track_id)
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **Spotify API** for audio feature data
- **scikit-learn** for machine learning algorithms
- **Streamlit** for the web interface
- **Plotly** for interactive visualizations

---

## 📞 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Happy listening! 🎵**# Spotify-Mood-Recommender
