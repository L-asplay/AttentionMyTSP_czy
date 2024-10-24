recode of changes : on  state 2
<font  size =1>no 缩放矩阵</font>

### target 
    想研究的问题是：地面端有很多数据处理的任务，任务之间有先后顺序关系，
    由于地面端处理能力有限，一部分对处理速度有要求的任务
    需要加载到带有服务器的无人机端处理，
    无人机首先选择其要加载过来的地面端任务，选择完之后
    就是之前研究的带有顺序约束的tsp问题，只不过是优化目标变成无人机的总能耗，
    然后每个任务有时间窗口，无人机应该在比时间窗口完成处理。文字描述大致就这样的。
    需要改代码的几个部分：
    首先是数据生成部分，这部分可以等同学写好再改；
    其次是无人机站选择哪些地面端任务处理，这个需要加个网络结构，
    先加个简单的全链接层试试；
    最后之前写的代码，要加时间窗口，改优化目标,mask激励。
    这周先试着把无人机选择任务的网络结构代码写写试试

## 无人机选择任务的网络结构代码 
### train.py
```py
from nets.task_selection.py import NodeSelector

def train_batch(
        model,
        optimizer,
        baseline,
        epoch,
        batch_id,
        step,
        batch,
        tb_logger,
        opts
):
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts.device)

    # 正常情况下 x 的形状 (batch_size, graph_size, node_dim)，直接投入使用
    # 现在假设 x是所有的地面任务 (batch_size, gena_size, node_dim) ，
    # 通过网络结构选择部分 变形为 (batch_size, graph_size, node_dim)
    # _, gena_size, node_dim = x.size()
    # node_selector = NodeSelector(gena_size, opts.graph_size, node_dim)
    # selected_nodes = node_selector(x)

    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    # Evaluate model, get costs and log probabilities
    cost, log_likelihood = model(x) # model(selected_nodes)

    # Evaluate baseline, get baseline loss if any (only for critic)
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)
                    # baseline.eval(selected_nodes, cost)

```

### task_selection.py
```py
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
```


