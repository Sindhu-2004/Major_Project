import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from data_loader import load_blogcatalog_data
from gcn_model import GCN
from isfo import detect_anomalies
from val import evaluate_anomalies 

def main():
    # Load dataset
    adj, features, labels = load_blogcatalog_data()

    # Convert to tensors
    adj = torch.tensor(adj.to_dense().numpy() if isinstance(adj, torch.Tensor) else adj, dtype=torch.float32)
    features = torch.tensor(features, dtype=torch.float32)

    # Train GCN
    device = torch.device("cpu")
    gcn = GCN(nfeat=features.shape[1], nhid=64, nout=32, dropout=0.5).to(device)
    optimizer = torch.optim.Adam(gcn.parameters(), lr=0.01)

    for epoch in range(50):
        gcn.train()
        optimizer.zero_grad()
        embeddings = gcn(features, adj)
        loss = torch.mean(embeddings)
        loss.backward()
        optimizer.step()

    embeddings = embeddings.detach().numpy()

    # Anomaly detection
    anomaly_labels = detect_anomalies(embeddings)  # -1 (anomaly), 1 (normal)

    pred_labels = np.where(anomaly_labels == -1, 1, 0)  # 1 = anomaly
    true_labels = np.where(labels == 0, 0, 1)  # assumes label 0 = normal, else anomaly

    metrics, conf_matrix = evaluate_anomalies(true_labels, pred_labels)

    print("\n--- Evaluation Metrics ---")
    for k, v in metrics.items():
        print(f"{k.capitalize()}: {v:.4f}")

    print("\n--- Confusion Matrix ---")
    print(conf_matrix)

    # Visualization
    cluster_labels = np.copy(anomaly_labels)
    num_anomalies = np.sum(anomaly_labels == -1)
    total_nodes = len(anomaly_labels)
    print(f"\nTotal Nodes: {total_nodes}, Anomalies Detected: {num_anomalies}")

    G = nx.from_numpy_array(adj.numpy())
    pos = nx.spring_layout(G, seed=42)

    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color="gray"), hoverinfo="none", mode="lines")

    node_x, node_y, node_color, node_text = [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        color = "red" if anomaly_labels[node] == -1 else "blue"
        node_color.append(color)
        node_text.append(f"Node: {node}<br>Cluster: {cluster_labels[node]}<br>Anomaly: {anomaly_labels[node]}")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers",
        marker=dict(size=10, color=node_color, line=dict(width=1, color="black")),
        text=node_text,
        hoverinfo="text"
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="Anomalies in BlogCatalog Dataset",
        showlegend=False,
        hovermode="closest",
        margin=dict(b=0, l=0, r=0, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )

    fig.show()

if __name__ == "__main__":
    main()
