import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from data_loader import load_blogcatalog_data
from gcn_model import GCN
from dbscan_clustering import dbscan_clustering
from isfo import detect_anomalies

def main():
    # Load dataset
    adj, features, labels = load_blogcatalog_data()

    # Convert adjacency matrix and features to PyTorch tensors
    adj = torch.tensor(adj.to_dense().numpy() if isinstance(adj, torch.Tensor) else adj, dtype=torch.float32)
    features = torch.tensor(features, dtype=torch.float32)

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

    embeddings = embeddings.detach().numpy()  # Convert to NumPy

    # Perform clustering
    cluster_labels = dbscan_clustering(embeddings)

    # Detect anomalies
    anomaly_labels = detect_anomalies(embeddings)

    # Count anomalies
    num_anomalies = np.sum(anomaly_labels == -1)
    total_nodes = len(anomaly_labels)
    print(f"Total Nodes: {total_nodes}, Anomalies Detected: {num_anomalies}")

    # Convert adjacency matrix to NetworkX graph
    G = nx.from_numpy_array(adj.numpy())  # Convert back to NumPy

    # Extract node positions using spring layout
    pos = nx.spring_layout(G, seed=42)  # Seed ensures consistent layout

    # Extract edges for plotting
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    # Create edges in plotly
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color="gray"),
        hoverinfo="none",
        mode="lines"
    )

    # Create nodes with hover info
    node_x = []
    node_y = []
    node_color = []
    node_text = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        color = "red" if anomaly_labels[node] == -1 else "blue"
        node_color.append(color)
        node_text.append(f"Node: {node}<br>Cluster: {cluster_labels[node]}<br>Anomaly: {anomaly_labels[node]}")

    # Create nodes in plotly
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers",
        marker=dict(
            size=10,
            color=node_color,
            line=dict(width=1, color="black")
        ),
        text=node_text,
        hoverinfo="text"
    )

    # Create final figure
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
