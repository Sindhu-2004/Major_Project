import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from data_loader import load_blogcatalog_data
from gcn_model import GCN
from dbscan_clustering import dbscan_clustering
from isfo import detect_anomalies

def main():
    # Load dataset
    adj, features, labels = load_blogcatalog_data()

    # Convert adjacency matrix & features to PyTorch tensors
    if isinstance(adj, np.ndarray):
        adj = torch.tensor(adj, dtype=torch.float32)
    elif isinstance(adj, torch.Tensor) and adj.is_sparse:
        adj = adj.to_dense()

    features = torch.tensor(features, dtype=torch.float32) if isinstance(features, np.ndarray) else features

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

    embeddings = embeddings.detach().cpu().numpy()  # Convert to numpy safely

    # Perform clustering
    cluster_labels = dbscan_clustering(embeddings)

    # Detect anomalies
    anomaly_labels = detect_anomalies(embeddings)

    # Count anomalies
    num_anomalies = np.sum(anomaly_labels == -1)
    total_nodes = len(anomaly_labels)
    print(f"Total Nodes: {total_nodes}, Anomalies Detected: {num_anomalies}")

    # Convert adjacency matrix to NetworkX graph
    adj_np = adj.cpu().numpy() if isinstance(adj, torch.Tensor) else adj
    G = nx.from_numpy_array(adj_np)

    # Extract only anomalous nodes
    anomaly_nodes = [i for i in range(total_nodes) if anomaly_labels[i] == -1]
    G_sub = G.subgraph(anomaly_nodes)  # Subgraph with anomalies

    # Use Kamada-Kawai layout for better visualization
    pos = nx.kamada_kawai_layout(G_sub)  

    # Plot anomalies with better visibility
    plt.figure(figsize=(10, 7))
    nx.draw(G_sub, pos, node_color='red', with_labels=False, node_size=50, edge_color="gray")
    plt.title("Anomalies in BlogCatalog Dataset (Graph Representation)")
    plt.show()

if __name__ == "__main__":
    main()
