import torch
import torch.nn as nn
import torch.nn.functional as F

class NodeSelector(nn.Module):
    def __init__(self, gena_size, graph_size, node_dim, hidden_dim=128, dropout=0.1):
        super(NodeSelector, self).__init__()
        self.gen_size = gena_size
        self.graph_size = graph_size

        # 用多层全连接网络进行特征变换
        self.fc1 = nn.Linear(node_dim, hidden_dim)  # 第一层全连接层
        self.fc2 = nn.Linear(hidden_dim, 1)  # 输出一个得分
        
        # Dropout层用于防止过拟合
        self.dropout = nn.Dropout(dropout)

        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # x 的形状: (batch_size, gena_size, node_dim)
        batch_size = x.size(0)
        
        # 全连接网络：提取每个节点的特征
        x = self.relu(self.fc1(x))  # 第一层全连接 + ReLU 激活
        x = self.dropout(x)  # Dropout
        # 计算每个节点的选择得分
        node_scores = self.fc2(x)  # 输出节点得分 (batch_size, gena_size, 1)
        
        # 通过 softmax 将得分转换为选择概率
        node_probs = F.softmax(node_scores, dim=1)  # (batch_size, gena_size, 1)

        # 根据选择概率进行采样，选择 graph_size 个节点
        selected_indices = torch.multinomial(node_probs.squeeze(-1), self.graph_size)  # (batch_size, graph_size)

        # 根据选中的索引从 x 中筛选节点
        batch_indices = torch.arange(batch_size).unsqueeze(-1).expand_as(selected_indices)  # (batch_size, graph_size)
        selected_nodes = x[batch_indices, selected_indices]  # (batch_size, graph_size, hidden_dim)

        return selected_nodes
    
# to be realized
#class AttNodeSelector(nn.Module):
#    def __init__(self):
#        super(NodeSelector, self).__init__()
#       
#    def forward(self, x):
#        return