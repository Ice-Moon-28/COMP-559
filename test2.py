import matplotlib.pyplot as plt

# 学习率（从大到小）
learning_rates = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5]
# 对应的值
values = [
    0.9993039898436512,
    0.9623706751232669,
    0.80835175312908255,
    0.51235829575625,
    0.13451235829575625,
    0.027000848118336213,
    0.0066898784188124237,
    0.0026898784188124237
]

# 反转顺序：从小到大
learning_rates_asc = learning_rates[::-1]
values_asc = values[::-1]

# 找到最大值的位置
max_index = values_asc.index(max(values_asc))
max_lr = learning_rates_asc[max_index]
max_val = values_asc[max_index]

# 画图
plt.figure(figsize=(8, 6))
plt.plot(learning_rates_asc, values_asc, marker='o', label='Value')

plt.xscale('log')  # 对数坐标
plt.xlabel('Tolerance level')
plt.ylabel('Accuracy')
plt.title('Accuracy in different Tolerance level')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()