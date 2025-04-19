# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv
# import torch.nn as nn

# from util.debug import debug_nan_in_tensor


# class EdgeGCNRegressor(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super().__init__()
#         self.gcn1 = GCNConv(in_channels, hidden_channels)
#         self.gcn2 = GCNConv(hidden_channels, out_channels)
#         self.edge_mlp = nn.Sequential(
#             nn.Linear(2 * out_channels, hidden_channels),
#             nn.ReLU(),
#             nn.Linear(hidden_channels, 1),
#             nn.Tanh(),
#         )

#         # # ✅ 初始化所有子模块权重
#         # self.gcn1.apply(self._init_weights)
#         # self.gcn2.apply(self._init_weights)
#         # self.edge_mlp.apply(self._init_weights)

#     def predict(self, x, edge_index):
#         src, tgt = edge_index
#         edge_input = torch.cat([x[src], x[tgt]], dim=1)
#         return self.edge_mlp(edge_input).squeeze(-1)

#     def forward(self, x, edge_index, edge_weight=None):
#         # debug_nan_in_tensor(x, edge_index, edge_weight)
#         x = self.gcn1(x, edge_index, edge_weight)

#         print("After gcn1, x contains nan:", torch.isnan(x).any())

#         # x = torch.nan_to_num(x, nan=0.0)
#         x = F.relu(x)
#         debug_nan_in_tensor(x, edge_index, edge_weight)
#         x = self.gcn2(x, edge_index, edge_weight)

#         print("After gcn2, x contains nan:", torch.isnan(x).any())
#         # x = torch.nan_to_num(x, nan=0.0)
#         debug_nan_in_tensor(x, edge_index, edge_weight)
#         x = F.relu(x)

#         return x
        

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.nn as nn

class EdgeGCNRegressor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # 两条独立路径：正边和负边
        self.gcn_pos_1 = GCNConv(in_channels, hidden_channels)
        self.gcn_neg_1 = GCNConv(in_channels, hidden_channels)
        self.gcn_pos_2 = GCNConv(hidden_channels, out_channels)
        self.gcn_neg_2 = GCNConv(hidden_channels, out_channels)

        # edge MLP 与之前一致
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * out_channels, 1 * hidden_channels),
            nn.ReLU(),
            nn.Linear(1 * hidden_channels, 1),
            # nn.Tanh(),
        )

    def predict(self, x, edge_index):
        src, tgt = edge_index
        edge_input = torch.cat([x[src], x[tgt]], dim=1)
        return self.edge_mlp(edge_input).squeeze(-1)

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is None:
            # 无边权时默认都为正边
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

        # 拆分正边和负边
        pos_mask = edge_weight >= 0
        neg_mask = edge_weight < 0

        pos_mask = pos_mask.squeeze()

        pos_edge_index = edge_index[:, pos_mask]
        pos_edge_weight = edge_weight[pos_mask]

        neg_mask = neg_mask.squeeze()

        neg_edge_index = edge_index[:, neg_mask]
        neg_edge_weight = edge_weight[neg_mask].abs()  # 注意取 abs！

        # 第一层：分别走正边和负边
        x_pos = self.gcn_pos_1(x, pos_edge_index, pos_edge_weight)
        x_neg = self.gcn_neg_1(x, neg_edge_index, neg_edge_weight)

        x = F.relu(x_pos - x_neg)  # 融合方式：差值，你也可以改成 x_pos + x_neg 或 concat

        # 第二层
        x_pos = self.gcn_pos_2(x, pos_edge_index, pos_edge_weight)
        x_neg = self.gcn_neg_2(x, neg_edge_index, neg_edge_weight)

        x = F.relu(x_pos - x_neg)

        return x