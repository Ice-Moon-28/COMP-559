import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import torch.nn as nn

class EdgeGATRegressor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, linear_hidden_channels=64, heads=4):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * out_channels, linear_hidden_channels),
            nn.ReLU(),
            nn.Linear(linear_hidden_channels, 1),
            nn.Tanh()
        )

    def predict(self, x, edge_index):
        src, tgt = edge_index
        edge_input = torch.cat([x[src], x[tgt]], dim=1)
        return self.edge_mlp(edge_input).squeeze(-1)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.gat1(x, edge_index)  # GAT 不接受 edge_weight
        x = F.relu(x)
        x = self.gat2(x, edge_index)
        x = F.relu(x)
        return x