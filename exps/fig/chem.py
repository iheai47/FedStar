import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def load_data(filepath):
    data = pd.read_csv(filepath)
    rounds = data['Step'].to_numpy()
    accuracy = data['Value'].to_numpy() * 100  # Convert to percentage
    return rounds, accuracy

# type = 'BIO-CHEM'
# type = 'BIO-SN-CV'
# type = 'BIO-CHEM-SN'
type = 'CHEM'
#FedAvg_modified.csv',   .csv',
data_mapping = {
    'Ours': './data/' + type + '/Ours.csv',
    'FedStar': './data/' + type + '/FedStar_modified.csv',
    'FedAvg': './data/' + type + '/FedAvg_modified.csv',
    'FedSage': './data/' + type + '/FedSage_modified.csv',
    'GCFL': './data/' + type + '/GCFL_modified.csv',
}

# Plotting
plt.figure(figsize=(10, 6))
dense_rounds = np.linspace(0, 200, 40)

# 遍历数据映射字典，载入数据并绘图
for label, filepath in data_mapping.items():
    rounds, accuracy = load_data(filepath)
    func = interp1d(rounds, accuracy, kind='quadratic', fill_value="extrapolate")
    plt.plot(dense_rounds, func(dense_rounds), label=label, linewidth=2.5)

# plt.yticks(np.arange(50, 100, 50))  # Adjust the range and step as needed
plt.xticks(np.arange(0, 201, 50))
ax = plt.gca()  # Get current axes

# Set limits for x and y axis to include zero
ax.set_xlim(left=0)
ax.set_ylim(bottom=60)  # Assuming 60 is a safe lower bound for y-axis

plt.xlabel('Communication Rounds', fontsize=20)
plt.ylabel('Avg Test Accuracy (%)', fontsize=20)
plt.tick_params(axis='both', labelsize=16)  # 设置x轴和y轴刻度标签大小为14

plt.legend(fontsize=16)  # 设置图例字体大小
plt.grid(True)

plt.savefig('./pdf/training_curve_' + type + '.pdf', format='pdf', bbox_inches='tight')

plt.show()
