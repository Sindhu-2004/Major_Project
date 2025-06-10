from sklearn.ensemble import IsolationForest
import numpy as np

def detect_anomalies(embeddings, contamination=0.05):
    """
    Detect anomalies using Isolation Forest.
    """
    model = IsolationForest(contamination=contamination, random_state=42)
    anomaly_labels = model.fit_predict(embeddings)
    return anomaly_labels
