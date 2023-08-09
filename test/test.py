import networkx as nx
from scipy.sparse import csr_matrix
from torch_geometric.utils import to_scipy_sparse_matrix
import networkx as nx
import torch

# 创建一个无向图
G = nx.Graph()
# 添加边
G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])

# 计算节点之间的最短距离
shortest_distances = dict(nx.all_pairs_shortest_path_length(G))
print(shortest_distances)

# 设置超参数 n_dg
n_dg = 10

# 生成节点的最短距离编码特征
node_features = []
for node in G.nodes():
    shortest_dist = shortest_distances[node]
    # 将节点的最短距离编码为长度为 n_dg 的向量，并进行 one-hot 编码
    dist_encoded = torch.zeros(n_dg)

    for dist in shortest_dist.values():
        dist_encoded[min(dist, n_dg - 1)] += 1.0
    node_features.append(dist_encoded)

# 将最短距离编码特征转换为 tensor
node_features_tensor = torch.stack(node_features)

# 输出节点的最短距离编码特征
print("Node Features (Shortest Distance Encoding):")
print(node_features_tensor)

#  cd /media/scdx/D/gdhe/FedStar/
#  conda activate FedStar

pred = torch.tensor([[0.1, 0.5, 0.3, 0.2, 0.4],
                     [0.7, 0.2, 0.9, 0.3, 0.6],
                     [0.5, 0.8, 0.6, 0.1, 0.9]])

print(pred.max(dim=1))
print(pred.max(dim=1)[1])
