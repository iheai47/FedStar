import matplotlib.pyplot as plt
import numpy as np
# 客户数量
clients = [7, 14, 28, 42, 56, 70, 84]

# 更新数据
fedavg_acc = [75, 75, 73.8, 70.5, 71, 73, 76]
gcfl_acc = [76, 74, 73.9, 73.5, 71, 72.6, 76]
fedstar_acc = [79.9, 77.3, 76.5, 76.4, 76, 77, 79.6]
ours_acc = [80, 78, 77, 77, 77, 78, 81]  # 新增的数据

# 绘制更新后的条形图，增大字号
fig, ax = plt.subplots(figsize=(14, 8))  # 调整图表大小以适应更多数据
# 设置色彩方案
# colors = plt.get_cmap('Set2')
colors = ['#FF5733', '#33C1FF', '#8D33FF', '#FFC733']  # 选用四种不同的鲜明颜色

# 计算条形图的位置
index = np.arange(len(clients))
width = 0.21  # 减小宽度以适应更多条形

fedavg_pos = index - 1.5 * width
gcfl_pos = index - 0.5 * width
fedstar_pos = index + 0.5 * width
ours_pos = index + 1.5 * width  # 新增位置

# 绘制条形图，使用颜色映射
# 绘制条形图
# 绘制条形图
ax.bar(fedavg_pos, fedavg_acc, width, label='FedAvg', color='#f1faee', edgecolor='black', linewidth=1.1, alpha=0.8)
ax.bar(gcfl_pos, gcfl_acc, width, label='GCFL', color='#a8dadc', edgecolor='black', linewidth=1.1, alpha=0.8)
ax.bar(fedstar_pos, fedstar_acc, width, label='FedStar', color='#457b9d', edgecolor='black', linewidth=1.1, alpha=0.8)
ax.bar(ours_pos, ours_acc, width, label='Ours', color='#e63946', edgecolor='black', linewidth=1.1, alpha=0.8)



# 添加标题和标签，增大字号
ax.set_xlabel('Num of Clients', fontsize=20)
ax.set_ylabel('Avg Test Acc (%)', fontsize=20)
ax.set_xticks(index)
ax.set_xticklabels(clients, fontsize=16)
ax.legend(fontsize=15)

# 设置y轴范围从0开始
ax.set_ylim(60, )
plt.savefig('./pdf/varying_client.pdf', format='pdf', bbox_inches='tight')

# 显示网格
# ax.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()
