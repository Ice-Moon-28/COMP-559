import torch
from torch_geometric.utils import degree

def debug_nan_in_tensor(x, edge_index, edge_weight=None, name="x", max_print=10):
    print(f"\n📊 Debugging tensor `{name}`...")

    has_nan = torch.isnan(x).any().item()
    has_inf = torch.isinf(x).any().item()
    print(f"  ➤ Contains NaN? {has_nan}")
    print(f"  ➤ Contains Inf? {has_inf}")

    if not has_nan and not has_inf:
        # print("✅ No NaNs or Infs detected.")
        # print("-" * 60)
        return

    # 找出 nan 或 inf 行
    nan_mask = torch.isnan(x).any(dim=1)
    inf_mask = torch.isinf(x).any(dim=1)
    bad_mask = nan_mask | inf_mask
    bad_indices = bad_mask.nonzero(as_tuple=True)[0]

    print(f"  ❗ Affected node indices: {bad_indices.tolist()[:max_print]} (total {bad_indices.numel()})")

    # 打印特征值统计
    print("  ➤ Node value stats:")
    for idx in bad_indices[:max_print].tolist():
        node_val = x[idx]
        print(f"    - Node {idx}: mean={node_val.mean().item():.4e}, std={node_val.std().item():.4e}, max={node_val.max().item():.4e}")

    # 计算度（以目标点为统计，即入度）
    deg = degree(edge_index[1], num_nodes=x.size(0))
    print("  ➤ Node degrees of affected nodes:")
    for idx in bad_indices[:max_print].tolist():
        print(f"    - Node {idx}: degree = {deg[idx].item():.1f}")

    # 打印边权（如果提供了 edge_weight）
    if edge_weight is not None:
        print("  ➤ Edge weights connected to NaN nodes:")
        src, tgt = edge_index
        for idx in bad_indices[:max_print].tolist():
            connected = ((src == idx) | (tgt == idx)).nonzero(as_tuple=True)[0]
            weights = edge_weight[connected].view(-1)
            if weights.numel() > 0:
                print(f"    - Node {idx}: edge_weight min={weights.min().item():.4e}, mean={weights.mean().item():.4e}, max={weights.max().item():.4e}")
            else:
                print(f"    - Node {idx}: ⚠️ No connected edges found!")

    print("-" * 60)