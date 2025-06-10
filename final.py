import torch
import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_blogcatalog_data
from gcn_model import GCN
from dbscan_clustering import dbscan_clustering
from isfo import detect_anomalies


def main():
    # Load dataset
    adj, features, labels = load_blogcatalog_data()

    # Define and train GCN
    device = torch.device("cpu")
    gcn = GCN(nfeat=features.shape[1], nhid=64, nout=32, dropout=0.5).to(device)
    optimizer = torch.optim.Adam(gcn.parameters(), lr=0.01)

    for epoch in range(50):  # Training loop
        gcn.train()
        optimizer.zero_grad()
        embeddings = gcn(features, adj)  # Extract node embeddings
        loss = torch.mean(embeddings)  # Placeholder loss
        loss.backward()
        optimizer.step()

    embeddings = embeddings.detach().numpy()  # Convert to numpy

    # Perform clustering
    cluster_labels = dbscan_clustering(embeddings)

    # Detect anomalies
    anomaly_labels = detect_anomalies(embeddings)

    # Count anomalies
    num_anomalies = np.sum(anomaly_labels == -1)
    total_nodes = len(anomaly_labels)
    print(f"Total Nodes: {total_nodes}, Anomalies Detected: {num_anomalies}")

    # Plot results
    plt.figure(figsize=(8, 6))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=anomaly_labels, cmap='coolwarm', edgecolors='k')
    plt.title("Anomaly Detection on BlogCatalog Dataset")
    plt.xlabel("Embedding Dimension 1")
    plt.ylabel("Embedding Dimension 2")
    plt.colorbar(label="Anomaly (-1) vs Normal (1)")
    plt.show()

if __name__ == "__main__":
    main()
