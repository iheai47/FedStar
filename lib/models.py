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




class GIN_ds(torch.nn.Module):
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
    def __init__(self, nfeat, n_se, nhid, nclass, nlayer, dropout, layer_num=2, n_pg = 16,):
        super(GIN_ds, self).__init__()
        self.num_layers = nlayer
        self.dropout = dropout
        self.layer_num = layer_num
        # self.anchorset_num = anchorset_num

        #hf0 将输入节点特征nfeat 转化为隐藏层维度nhid => hv0（输入嵌入）
        self.pre = torch.nn.Sequential(torch.nn.Linear(nfeat, nhid))
        # print("(输入节点特征=>隐藏层维度)",(nfeat, nhid))

        # self.conv_0 = torch.nn.Linear(anchorset_num, nhid) # 对锚集处理
        self.P_gnn_layer_1 = PGNN_layer(nhid, n_pg)
        # print("(PGNN_layer=>处理)", (nfeat, nhid - n_se))
        # if layer_num > 1:
        #     self.conv_hidden = torch.nn.ModuleList([PGNN_layer(nhid, nhid) for i in range(layer_num - 2)])
        #     # print("(PGNN_layer=>conv_hidden)", (nfeat, nhid - n_se))
        #     self.conv_out = PGNN_layer(nhid, n_pg)



        #fg0 将结构嵌入节点特征的维度n_se转换为隐藏层维度nhid => gv0 （输入嵌入）  args.n_se = args.n_rw + args.n_dg
        self.embedding_s = torch.nn.Linear(n_se, nhid)
        # print("(结构嵌入节点特征=>隐藏层维度)", (n_se, nhid))
        # self.pre_xP = torch.nn.Sequential(torch.nn.Linear(16, nhid))

        self.graph_convs = torch.nn.ModuleList()

        #将x 和 s 拼接后的维度转换为隐藏层维度nhid
        self.nn1 = torch.nn.Sequential(torch.nn.Linear(nhid + nhid, nhid), torch.nn.ReLU(), torch.nn.Linear(nhid, nhid))
        # print("将x 和 s 拼接后的维度转换为隐藏层维度nhid",(nhid + nhid, nhid))

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
        dists_max, dists_argmax = data.dists_max, data.dists_argmax
        # print(x.shape, "x.shape ")

        x = self.pre(x)
        x1 = x

        x_position, x1 = self.P_gnn_layer_1(x1, dists_max, dists_argmax)
        # print(dists_max.shape, "dists_max")
        # print(dists_argmax.shape, "dists_argmax")
        # print(x_position.shape, "x_position ")
        # print(x_structure.shape, "x_structure ")

        # for i in range(self.layer_num - 2):
        #     _, x1 = self.conv_hidden[i](x1, data.dists_max, data.dists_argmax)
        #     # x = F.relu(x) # Note: optional!
        #     if self.dropout:
        #         x1 = F.dropout(x1, training=self.training)
        # x_position, x1 = self.conv_out(x1, data.dists_max, data.dists_argmax)


        # x_position = F.normalize(x_position, p=2, dim=-1)

        s = torch.cat([s, x1], dim=1)
        # print(s.shape, "cat([s, x_position] 16+16")

        s = self.embedding_s(s)
        # print(s.shape, "embedding_ss经过结构化层后 32-> 64")
        # print(x.shape,"x.shape")
        # print(x_position.shape,"x_position.shape")

        # x2 = self.pre_xP(x_position) # 从16->64 在和结构嵌入拼接起来 通过一个线性层
        for i in range(len(self.graph_convs)):
            # x = torch.cat((x2, s), -1)
            x = torch.cat((x, s), -1)
            x = self.graph_convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            s = self.graph_convs_s_gcn[i](s, edge_index)
            s = torch.tanh(s)


        x = self.Whp(torch.cat((x, s), -1))
        x = global_add_pool(x, batch)
        x = self.post(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.readout(x)
        x = F.log_softmax(x, dim=1)

        return x

    def loss(self, pred, label):
        return F.nll_loss(pred, label)

class GIN_dc(torch.nn.Module):
    def __init__(self, nfeat, n_se, nhid, nclass, nlayer, dropout):
        super(GIN_dc, self).__init__()
        self.num_layers = nlayer
        self.dropout = dropout

        self.pre = torch.nn.Sequential(torch.nn.Linear(nfeat, nhid))

        self.embedding_s = torch.nn.Linear(n_se, nhid)

        self.graph_convs = torch.nn.ModuleList()
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



class PGNN_layer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, dist_trainable=True):
        super(PGNN_layer, self).__init__()
        self.input_dim = input_dim
        self.dist_trainable = dist_trainable

        if self.dist_trainable:
            self.dist_compute = Nonlinear(1, output_dim, 1)

        self.linear_hidden = torch.nn.Linear(input_dim*2, output_dim)
        self.linear_out_position = torch.nn.Linear(output_dim,1)
        self.act = torch.nn.ReLU()

        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                m.weight.data = torch.nn.init.xavier_uniform_(m.weight.data, gain=torch.nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = torch.nn.init.constant_(m.bias.data, 0.0)

    def forward(self, data_x, dists_max, dists_argmax):
        if self.dist_trainable:
            dists_max = self.dist_compute(dists_max.unsqueeze(-1)).squeeze()

        subset_features = data_x[dists_argmax.flatten(), :]
        subset_features = subset_features.reshape((dists_argmax.shape[0], dists_argmax.shape[1], data_x.shape[1]))
        messages = subset_features * dists_max.unsqueeze(-1)

        self_feature = data_x.unsqueeze(1).repeat(1, dists_max.shape[1], 1)
        messages = torch.cat((messages, self_feature), dim=-1)

        messages = self.linear_hidden(messages).squeeze()
        messages = self.act(messages) # n*m*d

        out_position = self.linear_out_position(messages).squeeze(-1)  # n*m_out
        out_structure = torch.mean(messages, dim=1)  # n*d
        # out_position 是每个节点相对于其锚点集的位置嵌入
        # 这个输出代表了节点在图中的位置信息。它是通过线性变换和聚合锚点集到节点的消息计算出来的，反映了节点相对于其锚点集的位置特性。

        # out_structure 是对 messages 在第二个维度（即锚点集维度）上的平均值，表示每个节点的结构信息
        # 这个输出捕获了节点的局部结构特征，不同于 out_position，它更多地关注于节点的局部邻域或其在图结构中的角色。
        '''
        首先，如果 dist_trainable 为真，则通过一个非线性层处理 dists_max，这可能用于学习节点间距离的表示。
        然后，从 feature 中选取子集特征，并结合节点间的距离信息 (dists_max)，形成 messages。
        接着，对 messages 应用线性变换和 ReLU 激活函数。
        最后，计算两种类型的输出：一种是基于位置的嵌入（out_position），另一种是基于结构的表示（out_structure）。
        这个过程体现了 P-GNN 模型的核心思想，即同时考虑图中节点的位置和结构特征。通过这种方式，P-GNN 能够更全面地捕获图中节点的复杂特性
        '''
        # print(dists_max.shape) #torch.Size([268, 16])
        # print(out_position.shape, out_structure.shape)
        # torch.Size([268, 16]) torch.Size([268, 64])
        return out_position, out_structure
        # return out_position
class Nonlinear(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Nonlinear, self).__init__()

        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, output_dim)

        self.act = torch.nn.ReLU()

        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                m.weight.data = torch.nn.init.xavier_uniform_(m.weight.data, gain=torch.nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = torch.nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x

class PGNN(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(PGNN, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if layer_num == 1:
            hidden_dim = output_dim
        if feature_pre:
            self.linear_pre = torch.nn.Linear(input_dim, feature_dim)
            self.conv_first = PGNN_layer(feature_dim, hidden_dim)
        else:
            self.conv_first = PGNN_layer(input_dim, hidden_dim)
        if layer_num>1:
            self.conv_hidden = torch.nn.ModuleList([PGNN_layer(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
            self.conv_out = PGNN_layer(hidden_dim, output_dim)

    def forward(self, data):
        x = data.x
        if self.feature_pre:
            x = self.linear_pre(x)
        x_position, x = self.conv_first(x, data.dists_max, data.dists_argmax)
        if self.layer_num == 1:
            return x_position
        # x = F.relu(x) # Note: optional!
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num-2):
            _, x = self.conv_hidden[i](x, data.dists_max, data.dists_argmax)
            # x = F.relu(x) # Note: optional!
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x_position, x = self.conv_out(x, data.dists_max, data.dists_argmax)
        x_position = F.normalize(x_position, p=2, dim=-1)
        return x_position