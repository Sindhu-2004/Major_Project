from sklearn.cluster import DBSCAN
import numpy as np

def dbscan_clustering(embeddings, eps=0.5, min_samples=5):
    """
    Perform DBSCAN clustering on node embeddings.
    """
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(embeddings)
    return clustering.labels_
