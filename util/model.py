import torch
import random
import numpy as np

def set_seed(seed=42):
    """
    设置所有相关随机种子，保证结果可复现（尽可能）
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多 GPU 情况下也设置

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 禁用加速算法的自动选择（保证可复现）

    print(f"✅ Random seed set to: {seed}")


def print_model_param_count(model, name="Model"):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"📊 {name} 参数统计：")
    print(f"  ➤ 总参数量      : {total_params:,}")
    print(f"  ➤ 可训练参数量  : {trainable_params:,}")