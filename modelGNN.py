import os

import networkx as nx
import torch
import pandas as pd
from pymongo import MongoClient
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
import json


# MongoDB Connection
MONGO_USER = "rootarded"
MONGO_PASSWORD = "<Password>"
MONGO_HOST = ""
MONGO_PORT = "27017"
DB_NAME_LOCAL = "EnergyLearningLocal"
DB_NAME = "EnergyLearning"
MONGO_URI_LOCAL = f"mongodb://localhost:{MONGO_PORT}/"
MONGO_URI = f"mongodb://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/{DB_NAME}"
SAVE_JSON_DIRECTORY = "../DataAnalysis/"


class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

def normalize_features(features):
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features)
    feature_names = ["avgRenewableRatio", "avgFossilFuelRatio", "weightedAvgTotalCo2Consumption", "weightedAvgTotalConsumption"]
    return torch.tensor(normalized_features, dtype=torch.float), scaler, feature_names

def train_gnn(graphs, epochs=200, lr=0.01):
    model = GNN(in_channels=4, hidden_channels=16, out_channels=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_values = []
    for epoch in range(epochs):
        total_loss = 0
        for data in graphs:
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = torch.mean(out ** 2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        loss_values.append(total_loss / len(graphs))
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Avg Loss {total_loss / len(graphs):.4f}")
    visualize_training(loss_values)
    return model


def get_embeddings(model, graphs):

    model.eval()
    embeddings_list = []
    with torch.no_grad():
        for data in graphs:
            embeddings = model(data.x, data.edge_index)
            embeddings_list.append(embeddings)
    return embeddings_list


def fetch_data():
    client = MongoClient(MONGO_URI_LOCAL)
    db = client[DB_NAME_LOCAL]
    collection = db['weightedZoneResultsv2']
    data = list(collection.find({}, {"_id": 0}))
    return pd.DataFrame(data).rename(columns={"zoneKey": "zone"})


def load_graph_data():
    client = MongoClient(MONGO_URI_LOCAL)
    db = client[DB_NAME_LOCAL]
    collection = db["ZoneBorders4"]
    return list(collection.find({}, {"_id": 0}))


def build_graph(zone_data, training_data):
    G = nx.Graph()

    training_map = {row["zone"]: row for _, row in training_data.iterrows()}

    for entry in zone_data:
        zone = entry["zoneKey"]

        if zone not in training_map:
            print(f"Warning: Zone {zone} not found in training_data. Skipping this zone.")
            continue


        zone_data = training_map[zone]


        missing_attrs = [
            attr for attr in [
                "avgRenewableRatio", "avgFossilFuelRatio",
                "weightedAvgTotalCo2Consumption", "weightedAvgTotalConsumption"
            ] if zone_data.get(attr) is None
        ]

        if missing_attrs:
            print(
                f"Warning: Zone {zone} is missing the following attributes: {', '.join(missing_attrs)}. Skipping this zone.")
            continue


        G.add_node(zone, **{
            "avgRenewableRatio": zone_data.get("avgRenewableRatio", 0),
            "avgFossilFuelRatio": zone_data.get("avgFossilFuelRatio", 0),
            "weightedAvgTotalCo2Consumption": zone_data.get("weightedAvgTotalCo2Consumption", 0),
            "weightedAvgTotalConsumption": zone_data.get("weightedAvgTotalConsumption", 0),
        })


        neighbors = entry.get("neighboring_zones", "").split(",") if entry.get("neighboring_zones") else []
        for neighbor in neighbors:
            neighbor = neighbor.strip()
            if neighbor:
                G.add_edge(zone, neighbor)
                G.add_edge(neighbor, zone)
    visualize_graph(G)
    return G



def visualize_training(loss_values):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=loss_values, mode='lines', name='Training Loss'))
    fig.update_layout(title='GNN Training Loss Over Epochs', xaxis_title='Epoch', yaxis_title='Loss')
    fig.show()


def visualize_embeddings(embeddings_list):
    embeddings = embeddings_list[0].numpy()
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=reduced[:, 0], y=reduced[:, 1], mode='markers', name='Embeddings'))
    fig.update_layout(title='GNN Embeddings Visualization (PCA)', xaxis_title='PCA1', yaxis_title='PCA2')
    fig.show()


def visualize_tsne(embeddings_list):
    embeddings = embeddings_list[0].numpy()
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced = tsne.fit_transform(embeddings)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=reduced[:, 0], y=reduced[:, 1], mode='markers', name='t-SNE Embeddings'))
    fig.update_layout(title='GNN Embeddings Visualization (t-SNE)', xaxis_title='t-SNE1', yaxis_title='t-SNE2')
    fig.show()

def visualize_umap(embeddings_list):
    embeddings = embeddings_list[0].numpy()
    reducer = umap.UMAP(n_components=2, n_jobs=-1)
    reduced = reducer.fit_transform(embeddings)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=reduced[:, 0], y=reduced[:, 1], mode='markers', name='UMAP Embeddings'))
    fig.update_layout(title='GNN Embeddings Visualization (UMAP)', xaxis_title='UMAP1', yaxis_title='UMAP2')
    fig.show()



def visualize_graph(G):

    pos = nx.spring_layout(G)

    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_y.append(y0)
        edge_y.append(y1)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=edge_x, y=edge_y,
                             line=dict(width=1, color='gray'),
                             hoverinfo='none', mode='lines'))

    fig.add_trace(go.Scatter(x=node_x, y=node_y,
                             mode='markers',
                             hoverinfo='text',
                             marker=dict(size=10, color='skyblue', line=dict(width=2, color='black')),
                             text=[f'Zone: {node}' for node in G.nodes()]))  # Hover text with zone name

    fig.update_layout(
        title='Interactive Graph Visualization',
        showlegend=False,
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        plot_bgcolor='white',
        height=800, width=800
    )

    fig.show()

def create_graphs():
    training_data = fetch_data()
    zone_data = load_graph_data()
    G = build_graph(zone_data, training_data)

    node_idx = {node: i for i, node in enumerate(G.nodes())}

    edge_index = torch.tensor(
        [[node_idx[u], node_idx[v]] for u, v in G.edges()], dtype=torch.long
    ).t().contiguous()

    raw_features = [
        [G.nodes[node].get(param, 0) for param in [
            "avgRenewableRatio", "avgFossilFuelRatio",
            "weightedAvgTotalCo2Consumption", "weightedAvgTotalConsumption"]]
        for node in G.nodes()
    ]

    x, scaler, feature_names = normalize_features(raw_features)

    return [Data(x=x, edge_index=edge_index)], [node_idx], scaler, feature_names


def detect_communities(graphs, node_maps, embeddings_list, min_co2, max_co2, scaler, feature_names):
    co2_index = feature_names.index("weightedAvgTotalCo2Consumption")  # Find correct index

    min_co2_scaled = scaler.transform([[0, 0, min_co2, 0]])[0][co2_index]
    max_co2_scaled = scaler.transform([[0, 0, max_co2, 0]])[0][co2_index]

    communities_per_graph = []

    for data, node_idx, embeddings in zip(graphs, node_maps, embeddings_list):
        node_usage = {node: 0 for node in node_idx}
        communities = []

        while node_usage:
            sorted_nodes = sorted(node_usage.keys(), key=lambda n: node_usage[n])
            start_node = sorted_nodes[0]

            community = {start_node}
            queue = [start_node]
            total_co2 = data.x[node_idx[start_node], co2_index].item()

            while queue:
                node = queue.pop(0)
                node_embedding = embeddings[node_idx[node]]

                similarities = {
                    neighbor: torch.dot(node_embedding, embeddings[node_idx[neighbor]]).item()
                    for neighbor in node_usage if neighbor not in community
                }
                sorted_neighbors = sorted(similarities.keys(), key=lambda n: similarities[n], reverse=True)

                for neighbor in sorted_neighbors:
                    neighbor_co2 = data.x[node_idx[neighbor], co2_index].item()
                    if total_co2 + neighbor_co2 > max_co2_scaled:
                        continue
                    community.add(neighbor)
                    total_co2 += neighbor_co2
                    queue.append(neighbor)
                    if min_co2_scaled <= total_co2 <= max_co2_scaled:
                        break

            if len(community) > 1:
                communities.append(community)

            for node in community:
                node_usage[node] += 1

            for node in community:
                node_usage.pop(node, None)

        communities_per_graph.append(communities)

    return communities_per_graph


def convert_sets_to_lists(communities):
    return [[list(community) for community in graph] for graph in communities]




def save_communities_to_json(communities_per_graph, directory=SAVE_JSON_DIRECTORY, filename="Groups.json"):
    try:

        communities_per_graph = convert_sets_to_lists(communities_per_graph)

        os.makedirs(directory, exist_ok=True)
        os.chdir(directory)

        with open(filename, 'w') as json_file:
            json.dump(communities_per_graph, json_file, indent=4)
        print(f"✅ Communities saved to {directory}/{filename}")
    except Exception as e:
        print(f"Error saving communities to JSON: {e}")


def main():
    min_co2, max_co2 = 3000000000, 5000000000
    print("📥 Recupero e elaborazione dei dati...")
    graphs, node_maps, scaler, feature_names = create_graphs()
    print("🧠 Allenamento del GNN...")
    model = train_gnn(graphs)
    print("🔍 Generazione degli embedding...")
    embeddings_list = get_embeddings(model, graphs)


    visualize_embeddings(embeddings_list)  # PCA
    visualize_tsne(embeddings_list)  # t-SNE
    visualize_umap(embeddings_list)  # UMAP

    print("🔎 Rilevamento delle comunità...")
    communities_per_graph = detect_communities(graphs, node_maps, embeddings_list, min_co2, max_co2, scaler, feature_names)

    for i, communities in enumerate(communities_per_graph):
        print(f"🌍 Grafo {i + 1} Comunità rilevate:", communities)

    save_communities_to_json(communities_per_graph)



if __name__ == "__main__":
    main()


