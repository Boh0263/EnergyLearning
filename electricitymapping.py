import pymongo
import networkx as nx
import plotly.graph_objects as go
import plotly.colors as pc


def fetch_data_from_mongodb(db_name, collection_name, uri="mongodb://localhost:27017/"):

    client = pymongo.MongoClient(uri)
    db = client[db_name]
    collection = db[collection_name]
    documents = list(collection.find())
    return documents


def create_graph(documents):

    G = nx.Graph()

    
    for doc in documents:
        node_id = doc.get("_id")
        lat = doc.get("center_1")
        lon = doc.get("center_0")

        
        if node_id and lat is not None and lon is not None:
            G.add_node(node_id, pos=(lon, lat))
            print(f"Node {node_id} added at position ({lon}, {lat})")
        else:
            print(f"Skipping node {node_id} due to missing coordinates")

    
    for doc in documents:
        node_id = doc.get("_id")
        neighboring_zones = doc.get("neighboring_zones", "")

        if neighboring_zones:
            
            neighboring_zones = neighboring_zones.split(",")
            for neighbor in neighboring_zones:
                if neighbor in G.nodes:
                    G.add_edge(node_id, neighbor)
                    print(f"Edge added between {node_id} and {neighbor}")
                else:
                    print(f"Skipping edge between {node_id} and {neighbor} (neighbor not found)")

    return G


def get_connected_components(G):

    connected_components = list(nx.connected_components(G))  
    subgraphs = [G.subgraph(component) for component in connected_components]  
    return subgraphs


def prepare_plot_data(subgraph, component_idx):

    positions = nx.get_node_attributes(subgraph, "pos")

    edge_x = []
    edge_y = []
    edge_text = []  
    node_x = []
    node_y = []
    node_text = []  
    node_color = []  

    for edge in subgraph.edges():
        node1, node2 = edge
        if node1 in positions and node2 in positions:
            x0, y0 = positions[node1]
            x1, y1 = positions[node2]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_text.append(f"Edge between {node1} and {node2}")  

    for node, (lon, lat) in positions.items():
        node_x.append(lon)
        node_y.append(lat)
        node_text.append(str(node))  
        node_color.append(component_idx)  

    return edge_x, edge_y, node_x, node_y, edge_text, node_text, node_color


def display_plot(subgraphs, num_components):

    fig = go.Figure()

    
    color_palette = pc.qualitative.Set1  

    
    if num_components > len(color_palette):
        color_palette = color_palette * (num_components // len(color_palette) + 1)  

    
    for i in range(num_components):
        edge_x, edge_y, node_x, node_y, edge_text, node_text, node_color = prepare_plot_data(subgraphs[i], i)

        
        component_color = color_palette[i]  

        
        fig.add_trace(go.Scatter(x=edge_x + node_x, y=edge_y + node_y,
                                 mode='markers+lines',
                                 marker=dict(size=10, color=[component_color]*len(node_x), colorscale='Viridis',
                                             line=dict(width=1, color='black')),
                                 text=edge_text + node_text,
                                 hoverinfo='text',
                                 name=f"Component {i+1}",
                                 visible=True))

    
    fig.update_layout(
        title="Geospatial Network Graph",
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        showlegend=True,  
        hovermode='closest',
        plot_bgcolor='rgb(10, 10, 10)',  
        paper_bgcolor='rgb(30, 30, 30)',  
        font=dict(color='white', size=14),
        margin=dict(l=0, r=0, t=50, b=0),  
    )

    
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='gray')
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='gray')

    fig.show()



def main():

    db_name = "EnergyLearningLocal"
    collection_name = "ZoneBorders3"

    
    documents = fetch_data_from_mongodb(db_name, collection_name)

    
    G = create_graph(documents)

    
    subgraphs = get_connected_components(G)

    
    num_components = len(subgraphs)
    display_plot(subgraphs, num_components)


if __name__ == "__main__":
    main()
