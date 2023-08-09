import torch
from torch_geometric.utils import to_networkx, degree, to_dense_adj, to_scipy_sparse_matrix
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from scipy import sparse as sp
import dgl
import numpy as np
import networkx as nx

def convert_to_nodeDegreeFeatures(graphs):
    graph_infos = []
    maxdegree = 0
    for i, graph in enumerate(graphs):
        g = to_networkx(graph, to_undirected=True)
        gdegree = max(dict(g.degree).values())
        if gdegree > maxdegree:
            maxdegree = gdegree
        graph_infos.append((graph, g.degree, graph.num_nodes))    # (graph, node_degrees, num_nodes)

    new_graphs = []
    for i, tuple in enumerate(graph_infos):
        idx, x = tuple[0].edge_index[0], tuple[0].x
        deg = degree(idx, tuple[2], dtype=torch.long)
        deg = F.one_hot(deg, num_classes=maxdegree + 1).to(torch.float)

        new_graph = tuple[0].clone()
        new_graph.__setitem__('x', deg)
        new_graphs.append(new_graph)

    return new_graphs

def get_maxDegree(graphs):
    maxdegree = 0
    for i, graph in enumerate(graphs):
        g = to_networkx(graph, to_undirected=True)
        gdegree = max(dict(g.degree).values())
        if gdegree > maxdegree:
            maxdegree = gdegree

    return maxdegree

def use_node_attributes(graphs):
    num_node_attributes = graphs.num_node_attributes
    new_graphs = []
    for i, graph in enumerate(graphs):
        new_graph = graph.clone()
        new_graph.__setitem__('x', graph.x[:, :num_node_attributes])
        new_graphs.append(new_graph)
    return new_graphs

def split_data(graphs, train=None, test=None, shuffle=True, seed=None):
    y = torch.cat([graph.y for graph in graphs])
    graphs_tv, graphs_test = train_test_split(graphs, train_size=train, test_size=test, stratify=y, shuffle=shuffle, random_state=seed)
    return graphs_tv, graphs_test


def get_numGraphLabels(dataset):
    s = set()
    for g in dataset:
        s.add(g.y.item())
    return len(s)


def _get_avg_nodes_edges(graphs):
    numNodes = 0.
    numEdges = 0.
    numGraphs = len(graphs)
    for g in graphs:
        numNodes += g.num_nodes
        numEdges += g.num_edges / 2.  # undirected
    return numNodes/numGraphs, numEdges/numGraphs


def get_stats(df, ds, graphs_train, graphs_val=None, graphs_test=None):
    df.loc[ds, "#graphs_train"] = len(graphs_train)
    avgNodes, avgEdges = _get_avg_nodes_edges(graphs_train)
    df.loc[ds, 'avgNodes_train'] = avgNodes
    df.loc[ds, 'avgEdges_train'] = avgEdges

    if graphs_val:
        df.loc[ds, '#graphs_val'] = len(graphs_val)
        avgNodes, avgEdges = _get_avg_nodes_edges(graphs_val)
        df.loc[ds, 'avgNodes_val'] = avgNodes
        df.loc[ds, 'avgEdges_val'] = avgEdges

    if graphs_test:
        df.loc[ds, '#graphs_test'] = len(graphs_test)
        avgNodes, avgEdges = _get_avg_nodes_edges(graphs_test)
        df.loc[ds, 'avgNodes_test'] = avgNodes
        df.loc[ds, 'avgEdges_test'] = avgEdges

    return df

def init_structure_encoding(args, gs, type_init):

    if type_init == 'rw':
        for g in gs:
            # Geometric diffusion features with Random Walk random walk-based structure embedding
            #将图数据转换为SciPy稀疏矩阵
            A = to_scipy_sparse_matrix(g.edge_index, num_nodes=g.num_nodes)
            D = (degree(g.edge_index[0], num_nodes=g.num_nodes) ** -1.0).numpy()

            Dinv=sp.diags(D)
            RW=A*Dinv
            M=RW

            SE_rw=[torch.from_numpy(M.diagonal()).float()]
            M_power=M
            for _ in range(args.n_rw-1):
                M_power=M_power*M
                SE_rw.append(torch.from_numpy(M_power.diagonal()).float())
            SE_rw=torch.stack(SE_rw,dim=-1)
            '''
            具体来看,SE_rw的行与列表示的意义:
            每一行代表一个节点的嵌入向量。SE_rw有N行,N为图的节点数量。
            每一列代表随机游走的步数。SE_rw有args.n_rw列,表示进行了args.n_rw步的随机游走。
            也就是说,取SE_rw中一个节点i的第t列,就是该节点执行t步随机游走后,结束节点的分布特征。
            
            举个例子:
            
            SE_rw[i, 3] 表示节点i进行3步随机游走,结束在各个节点上的概率分布。
            SE_rw[j, 5] 表示节点j进行5步随机游走,结束在各个节点上的概率分布。
            所以行是节点的索引,列是随机游走的步数。SE_rw封装了每个节点基于随机游走得到的结构信息。
            
            这种嵌入方式可以充分表达节点的STRUCTURAL CONTEXT,也就是节点在图网络拓扑结构中的位置信息。
            
            因此,SE_rw可以表示任意节点的拓扑嵌入特征,并作为图神经网络模型的有效输入之一。
            '''
            g['stc_enc'] = SE_rw

    elif type_init == 'dg':
        for g in gs:
            # PE_degree degree-based structure embedding
            g_dg = (degree(g.edge_index[0], num_nodes=g.num_nodes)).numpy().clip(1, args.n_dg)
            SE_dg = torch.zeros([g.num_nodes, args.n_dg])
            for i in range(len(g_dg)):
                SE_dg[i,int(g_dg[i]-1)] = 1

            g['stc_enc'] = SE_dg

    elif type_init == 'rw_dg':
        for g in gs:
            '''
            edge_index 是一个形状为 [2, 60] 的张量，表示图的边索引。每一列代表一条边，其中第一行是源节点的索引，第二行是目标节点的索引。
            总共有 60 条边。
            x 是一个形状为 [28, 53] 的张量，表示图的节点特征。每一行代表一个节点的特征向量，总共有 28 个节点，
            并且每个节点的特征向量维度为 53。
            y 是一个形状为 [1] 的张量，表示图的标签。在这个例子中，只有一个标签。
            综上所述，给定的数据表示一个包含 28 个节点、60 条边的图，每个节点具有一个 53 维的特征向量，并且有一个标签
            
            g.edge_index[0]
            
    tensor([[ 0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,  4,  4,  4,  5,  5,  5,
          6,  6,  6,  6,  7,  7,  7,  8,  8,  8,  8,  9,  9,  9, 10, 10, 10, 11,
         11, 11, 12, 13, 14, 14, 14, 15, 15, 15, 16, 16, 16, 17, 17, 17, 18, 18,
         18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
        [ 1,  5, 19,  0,  2, 13,  1,  3, 20,  2,  4,  6,  3,  5, 10,  0,  4, 21,
          3,  7, 11, 29,  6,  8, 30,  7,  9, 22, 23,  8, 10, 12,  4,  9, 31,  6,
         14, 18,  9,  1, 11, 15, 24, 14, 16, 25, 15, 17, 26, 16, 18, 27, 11, 17,
         28,  0,  2,  5,  8,  8, 14, 15, 16, 17, 18,  6,  7, 10]])

            g.edge_index 是一个形状为 (2, E) 的张量，其中 E 是边的数量。g.edge_index[0] 表示的是边的起始节点的索引信息，
            它是一个长度为 E 的一维张量，每个元素表示一条边的起始节点的索引。这个索引值可以对应于节点的编号或标识符，
            用于表示边连接的起始节点在节点列表中的位置。
            在这个包含两行的数组中，第1行与第2行中对应索引位置的值分别表示一条边的源节点和目标节点，LongTensor类型。
            
            
            Data(edge_index=[2, 22], x=[12, 38], edge_attr=[22, 3], y=[1])

            12个节点,每个节点特征长度为38
            22条边
            每条边有一个长度为3的特征向量
            整个图有一个标签,值为1
            
            '''
            # SE_rw
            A = to_scipy_sparse_matrix(g.edge_index, num_nodes=g.num_nodes)
            #将图形邻接矩阵作为SciPy 稀疏矩阵返回。
            #就是A就是邻接矩阵
            # print(" 将图形邻接矩阵作为SciPy 稀疏矩阵返回。")
            # print(g.edge_index)
            # print(A.todense())
            # print(" 将图形邻接矩阵作为SciPy 稀疏矩阵返回。")

            # print(degree(g.edge_index[0], num_nodes=g.num_nodes))
            #算了图中每个节点的度

            D = (degree(g.edge_index[0], num_nodes=g.num_nodes) ** -1.0).numpy()
            # print(" 计算了图中每个节点的度，并进行了倒数运算 ")
            # print(D)
            '''
            这行代码计算了图中每个节点的度，并进行了倒数运算。让我们逐步解释每个部分的含义：

            g.edge_index[0] 表示图中的边索引的第一个维度，它包含了图中所有边的起始节点。
            degree(g.edge_index[0], num_nodes=g.num_nodes) 是一个函数调用，
            它使用了g.edge_index[0]和num_nodes=g.num_nodes作为参数。这个函数用于计算图中每个节点的度，
            返回一个与图中节点数相同的一维张量。
            ** -1.0 是对度张量进行指数运算，将每个度值取倒数。这是为了得到归一化的度矩阵，使得度矩阵的对角线上的元素变为节点的度的倒数。
            .numpy() 是将度张量转换为NumPy数组，以便后续处理或计算。
            最终，变量D保存了图中每个节点度的倒数，作为归一化的度矩阵。
            '''

            Dinv=sp.diags(D)
            # print("用于创建一个稀疏对角矩阵，其中对角线元素由数组D的值给出")
            # print(Dinv)
            # 用于创建一个稀疏对角矩阵，其中对角线元素由数组D的值给出
            RW=A*Dinv
            # print("将矩阵A(将图形邻接矩阵作为SciPy 稀疏矩阵返回)与对角矩阵Dinv相乘。这里的A是表示图的邻接矩阵，Dinv是归一化的度矩阵的逆矩阵。")
            # print(RW)
            #，将矩阵A与对角矩阵Dinv相乘。这里的A是表示图的邻接矩阵，Dinv是归一化的度矩阵的逆矩阵。
            # 通过这个操作，我们实际上对邻接矩阵进行了归一化处理，以确保每个节点的邻居节点对其特征的贡献保持一致。
            M=RW
            #M表示经过随机游走结构嵌入（Random Walk-based Structure Embedding）处理后的邻接矩阵，
            # 它将用于后续的特征计算和图神经网络的训练过程。


            SE=[torch.from_numpy(M.diagonal()).float()]
            # print("将归一化邻接矩阵 M 的对角线元素转换为 PyTorch 的张量，并将其添加到 SE 列表中。")
            # print(SE)
            #将归一化邻接矩阵 M 的对角线元素转换为 PyTorch 的张量，并将其添加到 SE 列表中。
            M_power=M
            for _ in range(args.n_rw-1):
                M_power=M_power*M
                SE.append(torch.from_numpy(M_power.diagonal()).float())
                #M_power 乘以 M，相当于对归一化邻接矩阵进行了幂次操作。然后，将乘积结果的对角线元素转换为张量，并添加到 SE 列表中。

            # print(SE)
            SE_rw=torch.stack(SE,dim=-1)
            # print("随机游走的大小")
            # print(SE_rw.shape)
            # print(len(SE_rw))


            # PE_degree
            # print("计算图g中每个节点的度数")
            # print(degree(g.edge_index[0], num_nodes=g.num_nodes))
            g_dg = (degree(g.edge_index[0], num_nodes=g.num_nodes)).numpy().clip(1, args.n_dg)
            # print("numpy.clip()函数用于剪辑(限制)数组中的值")
            # print(g_dg)
            SE_dg = torch.zeros([g.num_nodes, args.n_dg])
            for i in range(len(g_dg)):
                SE_dg[i,int(g_dg[i]-1)] = 1
            # print("通过遍历每个节点的度数，将SE_dg中对应位置的元素设置为1，以实现独热编码。")
            # print(SE_dg)
            '''
                这段代码的作用是计算节点的度分布，并将其转换为一个独热编码的形式。
        首先，degree(g.edge_index[0], num_nodes=g.num_nodes)用于计算图g中每个节点的度数。
        g.edge_index是表示图中边连接关系的张量，g.edge_index[0]表示起始节点的索引，num_nodes=g.num_nodes指定节点的总数。
        degree()函数返回一个张量，其中每个元素表示对应节点的度数。
        然后，numpy().clip(1, args.n_dg)对度数张量进行处理。numpy()将度数张量转换为NumPy数组，
        numpy.clip()函数用于剪辑(限制)数组中的值。
        给定一个间隔，该间隔以外的值将被裁剪到间隔边。例如，如果指定间隔[0，16]，则小于0的值将变为0，而大于16的值将变为16。
        
        clip(1, args.n_dg)将度数限制在1到args.n_dg之间。这样做是为了确保度数在一个指定的范围内，通常用于限制特征的维度或范围。
        
        接下来，代码创建了一个大小为g.num_nodes x args.n_dg的全零张量SE_dg，用于存储独热编码形式的度分布。
        然后，通过遍历每个节点的度数，将SE_dg中对应位置的元素设置为1，以实现独热编码。
        
        总结起来，这段代码的目的是将图中每个节点的度分布转换为独热编码的形式，方便后续处理和分析
                '''

            g['stc_enc'] = torch.cat([SE_rw, SE_dg], dim=1)

    elif type_init == 'rw_sp':
        for g in gs:

            # # 创建一个空的无向图
            # G1 = nx.Graph()
            #
            # # 添加节点及其特征
            # for i, features in enumerate(g.x):
            #     G1.add_node(i, features=features)
            #
            # # 添加边
            # edges = g.edge_index.t().tolist()
            # G1.add_edges_from(edges)
            # print("无向图图像G:", G1)
            # print(g.edge_index)
            # print("原始图像g:",g)
            # print("有向图像:", G)

            # 创建图对象
            G = nx.DiGraph()
            # 添加节点和边
            for i in range(g.edge_index.shape[1]):
                source_node = int(g.edge_index[0, i])
                target_node = int(g.edge_index[1, i])
                G.add_edge(source_node, target_node)

            # 防止有的节点没有边未被添加
            if (g.num_nodes - G.number_of_nodes() != 0):
                for i in range(g.x.shape[0]):
                    G.add_node(i)


            # # 添加节点特征
            # for i in range(g.x.shape[0]):
            #     node_features = g.x[i:0].tolist()
            #     node_index = i
            #     G.nodes[node_index]['features'] = node_features

            # num_nodes = G.number_of_nodes()
            # # 初始化距离编码矩阵
            # distance_encoding = np.zeros((g.num_nodes, g.num_nodes))

            # 使用Floyd-Warshall算法计算节点之间的最短路径长度
            shortest_paths = dict(nx.all_pairs_shortest_path_length(G))
            #
            # 将最短路径长度编码为特征向量
            # for i in range(num_nodes):
            #     for j in range(num_nodes):
            #         distance_encoding[i][j] = shortest_paths[i][j]

            # 设置超参数 n_sp
            n_sp = args.n_sp

            # 生成节点的最短距离编码特征
            node_featuress = []
            # 将节点的最短距离编码为长度为 n_dg 的向量，并进行 one-hot 编码
            dist_encoded = torch.zeros(n_sp)

            for node in G.nodes():
                shortest_dist = shortest_paths[node]
                # print(node,shortest_dist)
                for dist in shortest_dist.values():
                    dist_encoded[min(dist, n_sp - 1)] += 1.0

                # arr = np.array()
                # # 计算总和
                # total = np.sum(arr)
                # # 每一项占总和的比例
                # ratio = arr / total

                total = sum(dist_encoded.clone())
                ratio = dist_encoded.clone() / total
                node_featuress.append(ratio)

                # 将最短距离编码特征转换为 tensor
            SE_sp = torch.stack(node_featuress)



            # SE_rw
            A = to_scipy_sparse_matrix(g.edge_index, num_nodes=g.num_nodes)
            D = (degree(g.edge_index[0], num_nodes=g.num_nodes) ** -1.0).numpy()
            # print(" 计算了图中每个节点的度，并进行了倒数运算 ")
            Dinv = sp.diags(D)
            # 用于创建一个稀疏对角矩阵，其中对角线元素由数组D的值给出
            RW = A * Dinv
            M = RW

            SE = [torch.from_numpy(M.diagonal()).float()]
            M_power = M
            for _ in range(args.n_rw - 1):
                M_power = M_power * M
                SE.append(torch.from_numpy(M_power.diagonal()).float())

            SE_rw = torch.stack(SE, dim=-1)


            g['stc_enc'] = torch.cat([SE_rw, SE_sp], dim=1)

    return gs


