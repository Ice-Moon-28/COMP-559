import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import PNAConv
from torch_geometric.nn import BatchNorm
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import degree


class EdgePNARegressor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, deg):
        super().__init__()

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.pna1 = PNAConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            aggregators=aggregators,
            scalers=scalers,
            deg=deg,
            towers=4,
            pre_layers=1,
            post_layers=1,
            divide_input=False
        )
        self.bn1 = BatchNorm(hidden_channels)

        self.pna2 = PNAConv(
            in_channels=hidden_channels,
            out_channels=out_channels,
            aggregators=aggregators,
            scalers=scalers,
            deg=deg,
            towers=4,
            pre_layers=1,
            post_layers=1,
            divide_input=False
        )
        self.bn2 = BatchNorm(out_channels)

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * out_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
            nn.Tanh(),
        )

    def predict(self, x, edge_index):
        src, tgt = edge_index
        edge_input = torch.cat([x[src], x[tgt]], dim=1)
        return self.edge_mlp(edge_input).squeeze(-1)

    def forward(self, x, edge_index, batch=None):
        x = self.pna1(x, edge_index)
        x = torch.nan_to_num(x, nan=0.0)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.pna2(x, edge_index)
        x = torch.nan_to_num(x, nan=0.0)
        x = self.bn2(x)
        x = F.relu(x)

        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)