import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, global_add_pool, SAGEConv
import random

#我们使用三层 GCN 作为结构编码器，使用三层 GIN 作为特征编码器，两者的隐藏大小均为 64

class serverGIN(torch.nn.Module):
    def __init__(self, nlayer, nhid):
        super(serverGIN, self).__init__()
        self.graph_convs = torch.nn.ModuleList()
        self.nn1 = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU(),
                                       torch.nn.Linear(nhid, nhid))
        self.graph_convs.append(GINConv(self.nn1))
        for l in range(nlayer - 1):
            self.nnk = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU(), torch.nn.Linear(nhid, nhid))
            self.graph_convs.append(GINConv(self.nnk))

class GIN(torch.nn.Module):
    #(Xu et al. 2019) 是一篇名为"How Powerful are Graph Neural Networks?"的论文，
    # 提出了一种基于图神经网络（GNN）的新型聚合函数GIN（Graph Isomorphism Network）
    def __init__(self, nfeat, nhid, nclass, nlayer, dropout):
        super(GIN, self).__init__()
        self.num_layers = nlayer
        self.dropout = dropout

        self.pre = torch.nn.Sequential(torch.nn.Linear(nfeat, nhid))

        self.graph_convs = torch.nn.ModuleList()
        self.nn1 = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU(), torch.nn.Linear(nhid, nhid))
        self.graph_convs.append(GINConv(self.nn1))
        for l in range(nlayer - 1):
            self.nnk = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU(), torch.nn.Linear(nhid, nhid))
            self.graph_convs.append(GINConv(self.nnk))

        self.post = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU())
        self.readout = torch.nn.Sequential(torch.nn.Linear(nhid, nclass))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.pre(x)
        for i in range(len(self.graph_convs)):
            x = self.graph_convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        x = global_add_pool(x, batch)
        x = self.post(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.readout(x)
        x = F.log_softmax(x, dim=1)
        return x

    def loss(self, pred, label):
        return F.nll_loss(pred, label)


class serverGIN_dc(torch.nn.Module):
    def __init__(self, n_se, nlayer, nhid):
        super(serverGIN_dc, self).__init__()

        self.embedding_s = torch.nn.Linear(n_se, nhid)
        self.Whp = torch.nn.Linear(nhid + nhid, nhid)

        self.graph_convs = torch.nn.ModuleList()
        self.nn1 = torch.nn.Sequential(torch.nn.Linear(nhid + nhid, nhid), torch.nn.ReLU(), torch.nn.Linear(nhid, nhid))
        self.graph_convs.append(GINConv(self.nn1))
        self.graph_convs_s_gcn = torch.nn.ModuleList()
        self.graph_convs_s_gcn.append(GCNConv(nhid, nhid))

        for l in range(nlayer - 1):
            self.nnk = torch.nn.Sequential(torch.nn.Linear(nhid + nhid, nhid), torch.nn.ReLU(), torch.nn.Linear(nhid, nhid))
            self.graph_convs.append(GINConv(self.nnk))
            self.graph_convs_s_gcn.append(GCNConv(nhid, nhid))

class GIN_dc(torch.nn.Module):
    # 我们使用三层 GCN 作为结构编码器，使用三层 GIN 作为特征编码器，两者的隐藏大小均为 64

    '''
    在模型的初始化方法__init__中，除了与之前相同的参数外，新增了一个参数n_se，表示辅助节点特征的维度。
    模型中添加了额外的组件用于处理辅助节点特征。

    模型的前向传播方法forward与之前的模型有一些不同的操作：

    在前向传播过程中，除了处理节点特征 x(节点特征) 外，还有辅助节点特征 s（结构嵌入）。
    在每一层的图卷积操作之前，将节点特征 x 和辅助节点特征 s 进行拼接，以使信息在节点特征和辅助节点特征之间传递。
    在每一层的图卷积操作之后，对辅助节点特征 s 进行GCN（Graph Convolutional Network）卷积操作，以更新辅助节点特征的表示。
    在最后一层的图卷积操作之后，将节点特征 x 和辅助节点特征 s 进行拼接，并经过一个线性层 self.Whp 进行处理。
    最后，通过全局池化操作（global_add_pool）将图中的节点特征进行汇总，然后通过一系列线性层和激活函数得到模型的输出结果，并通过F.log_softmax函数进行softmax归一化，得到类别概率分布。
    该模型通过引入辅助节点特征，在图分类任务中对节点特征和辅助节点特征进行联合学习，从而提升模型性能。前向传播方法中的操作充分利用了节点特征和辅助节点特征之间的交互信息。该模型同样定义了loss方法，用于计算模型预测结果与标签之间的损失（使用负对数似然损失函数）。
    通过使用这两个不同的模型，可以根据实际需求选择适合的模型来进行图分类任务。

    '''
    def __init__(self, nfeat, n_se, nhid, nclass, nlayer, dropout):
        super(GIN_dc, self).__init__()
        self.num_layers = nlayer
        self.dropout = dropout

        #hf0 将输入节点特征nfeat 转化为隐藏层维度nhid => hv0（输入嵌入）
        self.pre = torch.nn.Sequential(torch.nn.Linear(nfeat, nhid))
        # print("(输入节点特征=>隐藏层维度)",(nfeat, nhid))

        #fg0 将结构嵌入节点特征的维度n_se转换为隐藏层维度nhid => gv0 （输入嵌入）  args.n_se = args.n_rw + args.n_dg
        self.embedding_s = torch.nn.Linear(n_se, nhid)
        # print("(结构嵌入节点特征=>隐藏层维度)", (n_se, nhid))

        self.graph_convs = torch.nn.ModuleList()

        #将x 和 s 拼接后的维度转换为隐藏层维度nhid
        self.nn1 = torch.nn.Sequential(torch.nn.Linear(nhid + nhid, nhid), torch.nn.ReLU(), torch.nn.Linear(nhid, nhid))

        self.graph_convs.append(GINConv(self.nn1))
        self.graph_convs_s_gcn = torch.nn.ModuleList()
        self.graph_convs_s_gcn.append(GCNConv(nhid, nhid))

        for l in range(nlayer - 1):
            self.nnk = torch.nn.Sequential(torch.nn.Linear(nhid + nhid, nhid), torch.nn.ReLU(), torch.nn.Linear(nhid, nhid))
            self.graph_convs.append(GINConv(self.nnk))
            self.graph_convs_s_gcn.append(GCNConv(nhid, nhid))

        self.Whp = torch.nn.Linear(nhid + nhid, nhid)
        self.post = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU())
        self.readout = torch.nn.Sequential(torch.nn.Linear(nhid, nclass))

    def forward(self, data):
        #DataBatch(edge_index=[2, 8142], x=[3790, 37], y=[128], stc_enc=[3790, 32], batch=[3790], ptr=[129])
        x, edge_index, batch, s = data.x, data.edge_index, data.batch, data.stc_enc

        x = self.pre(x)
        s = self.embedding_s(s)

        for i in range(len(self.graph_convs)):
            x = torch.cat((x, s), -1)
            x = self.graph_convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            s = self.graph_convs_s_gcn[i](s, edge_index)
            s = torch.tanh(s)
            # s = torch.relu (s)
            # s = torch.sigmoid(s)

        x = self.Whp(torch.cat((x, s), -1))
        x = global_add_pool(x, batch)
        x = self.post(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.readout(x)
        x = F.log_softmax(x, dim=1)
        return x

    def loss(self, pred, label):
        return F.nll_loss(pred, label)


class serverGraphSage(torch.nn.Module):
    def __init__(self, nlayer, nhid):
        super(serverGraphSage, self).__init__()
        self.graph_convs = torch.nn.ModuleList()
        self.graph_convs.append(SAGEConv(nhid, nhid))
        for l in range(nlayer - 1):
            self.graph_convs.append(SAGEConv(nhid, nhid))

class GraphSage(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayer, dropout):
        super(GraphSage, self).__init__()
        self.num_layers = nlayer
        self.dropout = dropout

        self.pre = torch.nn.Sequential(torch.nn.Linear(nfeat, nhid))

        self.graph_convs = torch.nn.ModuleList()
        self.graph_convs.append(SAGEConv(nhid, nhid))

        for l in range(nlayer - 1):
            self.graph_convs.append(SAGEConv(nhid, nhid))

        self.post = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU())
        self.readout = torch.nn.Sequential(torch.nn.Linear(nhid, nclass))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.pre(x)
        for i in range(len(self.graph_convs)):
            x = self.graph_convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        x = global_add_pool(x, batch)
        x = self.post(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.readout(x)
        x = F.log_softmax(x, dim=1)
        return x

    def loss(self, pred, label):
        return F.nll_loss(pred, label)