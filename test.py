import matplotlib.pyplot as plt
import pandas as pd

# 数据定义：随机种子 + 四个任务的数据
data = {
    "Random Seed": [42, 888, 1228, 3047],
    "MNIST": [0.00500225, 0.0052195, 0.0053195, 0.00518175],
    "FASHION-MNIST": [0.00746075, 0.00726034, 0.00716012, 0.0069895],
}

# 创建 DataFrame
df = pd.DataFrame(data)

# 设置索引为随机种子（横轴）
df.set_index("Random Seed").plot(kind="bar", figsize=(10, 6))

# 图表标题和轴标签
plt.title("Validation Mean Accuracy of MLP Models on Similiar Tasks")
plt.xlabel("Random Seed")
plt.ylabel("Validation Accuracy")

# 设置 y 轴范围
plt.ylim(0.85, 1.0)

# 网格和图例
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc='upper left')

# 自动布局
plt.tight_layout()

# 显示图表
plt.show()