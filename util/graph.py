import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from timm import create_model
import matplotlib.pyplot as plt
import networkx as nx
from torch.utils.data import DataLoader
from torch_geometric.utils import degree

from util.debug import debug_nan_in_tensor

def add_self_loop(node_features, edge_index, edge_attr):
    in_deg = degree(edge_index[1], num_nodes=len(node_features))

    isolated_nodes = (in_deg == 0).nonzero(as_tuple=True)[0]

    print(f"🔗 Number of isolated nodes: {isolated_nodes.numel()}")
    num_isolated = isolated_nodes.numel()

    if num_isolated > 0:
        print(f"🔗 Adding self-loops to {num_isolated} isolated nodes...")

        # 添加自环索引和边权
        self_loops = torch.stack([isolated_nodes, isolated_nodes], dim=0)
        self_loop_attr = torch.full((num_isolated, 1), 0.1, dtype=torch.float)

        # 拼接到原始图上
        edge_index = torch.cat([edge_index, self_loops], dim=1)
        edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)

    return edge_index, edge_attr

def extract_neuron_level_graph_with_input_layer(model, layer_limit=3):
    edge_index = []
    edge_attr = []
    node_features = []
    node_id = 0
    linear_layer_count = 0
    layer_infos = []
    input_nodes = []
    last_layer_nodes = []

    # 获取所有 Linear 层信息
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_layer_count += 1
            if linear_layer_count > layer_limit:
                break
            layer_infos.append((name, module))

    total_layers = len(layer_infos)
    if total_layers == 0:
        raise ValueError("No linear layers found in model.")

    
    first_linear = layer_infos[0][1]
    input_size = first_linear.in_features
    for i in range(input_size):
        # 可以给输入节点一些基础特征，例如位置、层编号为 -1
        feature = [
            0.0,
            0.0,
            0.0,
            0,
            0,
            -1.0,
            i / (input_size - 1) if input_size > 1 else 0.0,
        ]
        node_features.append(feature)
        input_nodes.append(node_id)
        node_id += 1

    last_layer_nodes = input_nodes  # 初始化为输入层神经元

    # === 正常处理 Linear 层 ===
    for layer_id, (name, module) in enumerate(layer_infos):
        W = module.weight.data.cpu()  # [out_features, in_features]
        fan_out, fan_in = W.size()
        layer_nodes = []

        for i, row in enumerate(W):  # 遍历当前层的每个 output neuron
            mean = row.mean().item()
            std = row.std().item()
            layer_id_normalized = layer_id / (total_layers - 1) if total_layers > 1 else 0.0
            neuron_pos = i / (fan_out - 1) if fan_out > 1 else 0.0
            bias = module.bias.data[i].item() if module.bias is not None else 0.0

            feature = [
                bias,
                mean,
                std,
                fan_in,
                fan_out, 
                layer_id_normalized,
                neuron_pos,
            ]
            node_features.append(feature)
            layer_nodes.append(node_id)
            node_id += 1

        # 连接上一层 → 当前层，生成 edge 和 edge_attr
        for src_idx, src in enumerate(last_layer_nodes):
            for tgt_idx, tgt in enumerate(layer_nodes):
                edge_index.append([src, tgt])
                weight = W[tgt_idx, src_idx].item()  # W[out, in]
                edge_attr.append([weight])

        last_layer_nodes = layer_nodes  # 当前层变成下一轮的 last

    # 转换为 Tensor 格式
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float) if edge_attr else torch.empty((0, 1), dtype=torch.float)
    x = torch.tensor(node_features, dtype=torch.float)

    edge_index, edge_attr = add_self_loop(x, edge_index, edge_attr)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def extract_neuron_level_graph_without_input_layer(model, layer_limit=3):
    edge_index = []
    edge_attr = []
    node_features = []
    node_id = 0
    last_layer_nodes = []
    linear_layer_count = 0

    layer_infos = []

    # 记录每层 Linear 层的信息
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_layer_count += 1
            if linear_layer_count > layer_limit:
                break
            layer_infos.append((name, module))

    total_layers = len(layer_infos)

    for layer_id, (name, module) in enumerate(layer_infos):
        W = module.weight.data.cpu()  # shape: (out_features, in_features)
        fan_out, fan_in = W.size()
        layer_nodes = []

        for i, row in enumerate(W):  # row: weights for neuron i in current layer
            mean = row.mean().item()
            std = row.std().item()

            bias = module.bias.data[i].item() if module.bias is not None else 0.0

            layer_id_normalized = layer_id / (total_layers - 1) if total_layers > 1 else 0.0
            neuron_pos = i / (fan_out - 1) if fan_out > 1 else 0.0

            feature = [
                mean,
                std,
                bias,
                # fan_in,
                # fan_out,
                layer_id_normalized,
                neuron_pos,
            ]
            node_features.append(feature)
            layer_nodes.append(node_id)
            node_id += 1

        # 构造边和边的权重
        if last_layer_nodes:
            for src_idx, src in enumerate(last_layer_nodes):  # input neuron index
                for tgt_idx, tgt in enumerate(layer_nodes):  # output neuron index
                    edge_index.append([src, tgt])
                    weight = W[tgt_idx, src_idx].item()  # 注意维度：W[output, input]
                    edge_attr.append([weight])  # 用 [] 保持 shape 为 (N, 1)

        last_layer_nodes = layer_nodes

    if not edge_index:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
        # .abs()
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        # .abs()

  

    x = torch.tensor(node_features, dtype=torch.float)
   
    edge_index, edge_attr = add_self_loop(x, edge_index, edge_attr)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def split_edges(x, edge_index, edge_attr, split_ratio=0.8, seed=42):
    assert edge_index.size(1) == edge_attr.size(0), "edge count mismatch"

    num_edges = edge_index.size(1)
    indices = torch.randperm(num_edges, generator=torch.Generator().manual_seed(seed))

    split_point = int(num_edges * split_ratio)
    train_idx = indices[:split_point]
    valid_idx = indices[split_point:]

    train_edge_index = edge_index[:, train_idx]
    train_edge_attr = edge_attr[train_idx]

    valid_edge_index = edge_index[:, valid_idx]
    valid_edge_attr = edge_attr[valid_idx]

    return Data(x=x, edge_index=train_edge_index, edge_attr=train_edge_attr), Data(x=x, edge_index=valid_edge_index, edge_attr=valid_edge_attr)