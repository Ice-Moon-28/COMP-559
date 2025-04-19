import matplotlib.pyplot as plt
import pandas as pd

# 数据定义
data = {
    "Random Seed": [42, 888, 1228, 3047],
    "MNIST": [0.00500225, 0.0052195, 0.0053195, 0.00518175],
    "FASHION-MNIST": [0.00746075, 0.00726034, 0.00716012, 0.0069895],
}

df = pd.DataFrame(data)
df.set_index("Random Seed", inplace=True)

# 设置颜色
custom_colors = ['#1f77b4', '#ff7f0e']  # 蓝色 + 橙色

# 创建柱状图
ax = df.plot(kind="bar", figsize=(10, 6), color=custom_colors)

# 设置标题、坐标轴标签、y轴范围
plt.title("Validation Accuracy of MLP Models on Similar Tasks")
plt.xlabel("Random Seed")
plt.ylabel("Validation Accuracy")
plt.ylim(0.0045, 0.0076)

# 网格和图例
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc='upper left')

# 自动布局
plt.tight_layout()
plt.show()