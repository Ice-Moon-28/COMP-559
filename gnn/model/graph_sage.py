import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import torch.nn as nn


class EdgeGraphSAGERegressor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.sage1 = SAGEConv(in_channels, hidden_channels)
        self.sage2 = SAGEConv(hidden_channels, out_channels)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * out_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
            nn.Tanh(),
        )

        # # ✅ 初始化权重（可选）
        # self.sage1.apply(self._init_weights)
        # self.sage2.apply(self._init_weights)
        # self.edge_mlp.apply(self._init_weights)

    def predict(self, x, edge_index):
        src, tgt = edge_index
        edge_input = torch.cat([x[src], x[tgt]], dim=1)
        return self.edge_mlp(edge_input).squeeze(-1)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.sage1(x, edge_index)
        x = torch.nan_to_num(x, nan=0.0)
        x = F.relu(x)

        x = self.sage2(x, edge_index)
        x = torch.nan_to_num(x, nan=0.0)
        x = F.relu(x)

        return x

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, SAGEConv)):
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.xavier_uniform_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)