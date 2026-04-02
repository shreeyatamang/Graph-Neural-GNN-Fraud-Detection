import pandas as pd
import torch
from torch_geometric.data import Data


def load_clean_data(path):
    df = pd.read_csv(path)
    return df


def build_graph(df):
    print("Building graph...")

    nodes = list(set(df['nameOrig']).union(set(df['nameDest'])))
    node_map = {node: i for i, node in enumerate(nodes)}
    num_nodes = len(nodes)
    print(f"Total nodes: {num_nodes}")

    node_features = torch.zeros((num_nodes, 12))

   
    sender = df.groupby('nameOrig').agg(
        tx_count        = ('amount', 'count'),
        avg_amount      = ('amount', 'mean'),
        max_amount      = ('amount', 'max'),
        total_amount    = ('amount', 'sum'),
        std_amount      = ('amount', 'std'),
        unique_dest     = ('nameDest', 'nunique'), 
        step_min        = ('step', 'min'),
        step_max        = ('step', 'max'),
    ).reset_index()
    sender['std_amount']   = sender['std_amount'].fillna(0)
    sender['time_span']    = sender['step_max'] - sender['step_min']
    sender['dest_ratio']   = sender['unique_dest'] / sender['tx_count'] 

    for _, row in sender.iterrows():
        idx = node_map[row['nameOrig']]
        node_features[idx, 0]  = row['tx_count']
        node_features[idx, 1]  = row['avg_amount']
        node_features[idx, 2]  = row['max_amount']
        node_features[idx, 3]  = row['total_amount']
        node_features[idx, 4]  = row['std_amount']
        node_features[idx, 5]  = row['unique_dest']
        node_features[idx, 6]  = row['time_span']
        node_features[idx, 7]  = row['dest_ratio']

    
    receiver = df.groupby('nameDest').agg(
        recv_count      = ('amount', 'count'),
        recv_avg        = ('amount', 'mean'),
        recv_max        = ('amount', 'max'),
        unique_senders  = ('nameOrig', 'nunique'),  
    ).reset_index()

    for _, row in receiver.iterrows():
        if row['nameDest'] in node_map:
            idx = node_map[row['nameDest']]
            node_features[idx, 8]  = row['recv_count']
            node_features[idx, 9]  = row['recv_avg']
            node_features[idx, 10] = row['recv_max']
            node_features[idx, 11] = row['unique_senders']

   
    for col in range(12):
        col_max = node_features[:, col].max()
        if col_max > 0:
            node_features[:, col] /= col_max

    
    edge_index  = []
    edge_labels = []

    for _, row in df.iterrows():
        src = node_map[row['nameOrig']]
        dst = node_map[row['nameDest']]
        edge_index.append([src, dst])
        edge_labels.append(row['isFraud'])

    edge_index  = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_labels = torch.tensor(edge_labels, dtype=torch.long)

    data = Data(x=node_features, edge_index=edge_index)
    print(f"Graph built! Node feature shape: {data.x.shape}")

    return data, edge_labels


if __name__ == "__main__":
    df = load_clean_data("data/cleaned_paysim.csv")

    fraud_df  = df[df['isFraud'] == 1]
    normal_df = df[df['isFraud'] == 0]

    print(f"Total fraud rows available: {len(fraud_df)}")
    print(f"Total normal rows available: {len(normal_df)}")

    normal_sample = normal_df.sample(
        n=min(len(fraud_df) * 10, len(normal_df)),
        random_state=42
    )

    df = pd.concat([fraud_df, normal_sample]).sample(frac=1, random_state=42)
    print(f"Balanced sample size: {len(df)} rows")
    print(f"Fraud ratio: {df['isFraud'].mean():.2%}")

    data, edge_labels = build_graph(df)

    print(data)
    torch.save(data, "data/graph.pt")
    torch.save(edge_labels, "data/edge_labels.pt")
    print("Edge labels shape:", edge_labels.shape)