import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGEFraud(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim // 2)

       
        edge_input_dim = hidden_dim // 2 * 2
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv3(x, edge_index)
        x = F.relu(x)

        return x

    def predict_edges(self, x, edge_index):
        node_emb = self.forward(x, edge_index)

        src = edge_index[0]
        dst = edge_index[1]

        
        edge_emb = torch.cat([node_emb[src], node_emb[dst]], dim=1)
        return self.edge_mlp(edge_emb).squeeze(1)