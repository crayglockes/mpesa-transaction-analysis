"""
segmentation.py
---------------
Reusable wrappers around the K-Means segmentation logic in
notebooks/03_user_segmentation.ipynb.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def find_optimal_k(X_scaled: np.ndarray, k_range=range(2, 11)) -> int:
    """Return the k that maximises silhouette score."""
    scores = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        scores.append(silhouette_score(X_scaled, km.fit_predict(X_scaled), sample_size=2000))
    return list(k_range)[int(np.argmax(scores))]


def fit_kmeans(X_scaled: np.ndarray, k: int) -> KMeans:
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    km.fit(X_scaled)
    return km


def pca_2d(X_scaled: np.ndarray):
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(X_scaled), pca
