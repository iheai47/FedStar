import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class DistanceEncodingModule(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(DistanceEncodingModule, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, encoding_dim),  # 2 features: source node and target node distances
            nn.ReLU(),
            nn.Linear(encoding_dim, encoding_dim)
        )

    def forward(self, distances):
        encoded_distances = self.encoder(distances)
        return encoded_distances


class GCNWithDistanceEncoding(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, encoding_dim):
        super(GCNWithDistanceEncoding, self).__init__()
        self.gcn_conv = GCNConv(input_dim, hidden_dim)
        self.distance_encoder = DistanceEncodingModule(input_dim=2, encoding_dim=encoding_dim)
        self.readout = nn.Linear(hidden_dim + encoding_dim, output_dim)

    def forward(self, x, edge_index, batch, distances):
        x = self.gcn_conv(x, edge_index)
        encoded_distances = self.distance_encoder(distances)
        x_with_distance = torch.cat((x, encoded_distances), dim=1)
        x = self.readout(x_with_distance)
        return F.log_softmax(x, dim=1)


# 创建模型实例
input_dim = 16  # 节点特征维度
hidden_dim = 64  # 隐藏层维度
output_dim = 2  # 输出类别数
encoding_dim = 32  # 距离编码维度
model = GCNWithDistanceEncoding(input_dim, hidden_dim, output_dim, encoding_dim)

# 构造数据并进行训练
# 假设有 x (节点特征), edge_index (边索引), batch (节点批次), distances (节点间距离)
x = torch.randn(100, input_dim)
edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)  # 临接矩阵的边索引
batch = torch.tensor([0, 0, 1, 2], dtype=torch.long)  # 节点的批次信息
distances = torch.randn(100, 2)  # 假设每个节点有两个距离特征
labels = torch.tensor([0, 1, 0])  # 假设有三个类别

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(10):
    model.train()
    optimizer.zero_grad()
    output = model(x, edge_index, batch, distances)
    loss = F.nll_loss(output, labels)
    loss.backward()
    optimizer.step()

# 在测试集上进行测试
model.eval()
with torch.no_grad():
    test_output = model(x, edge_index, batch, distances)
    predicted_labels = test_output.argmax(dim=1)
