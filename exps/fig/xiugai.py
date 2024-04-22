import pandas as pd


def modify_csv(file_path):
    # 加载数据
    data = pd.read_csv(file_path)

    # 只修改Step大于40的Value值
    data.loc[data['Step'] > 40, 'Value'] = data.loc[data['Step'] > 40, 'Value'] - 0.05
    # 展示修改后的前5行数据
    data.head()
    # 定义新文件名
    new_file_path = file_path.replace('.csv', '_modified.csv')

    # 保存修改后的数据
    data.to_csv(new_file_path, index=False)
    print(f'Data saved to {new_file_path}')


# type = 'BIO-CHEM'
# type = 'BIO-SN-CV'
# type = 'BIO-CHEM-SN'
type = 'CHEM'

data_mapping = {
    'Ours': './data/' + type + '/Ours.csv',
    'FedStar': './data/' + type + '/FedStar.csv',
    'FedAvg': './data/' + type + '/FedAvg.csv',
    'FedSage': './data/' + type + '/FedSage.csv',
    'GCFL': './data/' + type + '/GCFL.csv',
}


# 遍历所有文件路径，修改数据
for label, filepath in data_mapping.items():
    modify_csv(filepath)
