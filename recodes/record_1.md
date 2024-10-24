record of changes

### 1. pretrained/tsp_20/args.json
1. 添加新参数 order_size 
    每次指定生成点队列的最后 order_size个有优先级顺序
2. 添加新参数 lr_encode
    在 attention 实现优先偏置 W_r 的参数
    ~ $C^{eln(\frac{o}{g}+1)}$
3. 添加新参数 sub_encode_layers
    设置 sub_encode 的堆叠层数

epoch_size: 1280000 -> 128000

### 2. options.py
1. model方面,添加新的 args 设置

### 3. nets/attention_model.py/class AttentionModel
1. __init__()
```py
    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 problem,
                 n_encode_layers=2,

                 order_size=0,
                 lr_encode=1.0,
                 sub_encode_layers=0,

                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='batch',
                 n_heads=8,
                 checkpoint_encoder=False,
                 shrink_size=None):
        super(AttentionModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers

        self.order_size=order_size
        self.lr_encode=lr_encode
        self.sub_encode_layers=sub_encode_layers

        self.decode_type = None
        self.temp = 1.0
        self.allow_partial = problem.NAME == 'sdvrp'
        self.is_vrp = problem.NAME == 'cvrp' or problem.NAME == 'sdvrp'
        self.is_orienteering = problem.NAME == 'op'
        self.is_pctsp = problem.NAME == 'pctsp'

        self.tanh_clipping = tanh_clipping

        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        self.problem = problem
        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size

        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,

            order_size=self.order_size,
            lr_encode=self.lr_encode,
            sub_layers=self.sub_encode_layers,

            normalization=normalization
        )
        # Remianing

```
2. Remaining: decode 中的 attention 维持原来的方案
   
### 4. nets/graph_encoder.py
1. class GraphAttentionEncoder
```py
class GraphAttentionEncoder(nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            n_layers,

            order_size,
            lr_encode,
            sub_layers,

            node_dim=None,
            normalization='batch',
            feed_forward_hidden=512
    ):
        super(GraphAttentionEncoder, self).__init__()

        # To map input to embedding space
        self.init_embed = nn.Linear(node_dim, embed_dim) if node_dim is not None else None
        
        self.order_size = order_size

        self.layers1 = nn.Sequential(*(
            MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization)
            for _ in range(sub_layers)
        ))
        
        self.layers2 = nn.Sequential(*(
            MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization,
            order_size,lr_encode)
            for _ in range(sub_layers)
       ))

        self.layers = nn.Sequential(*(
            MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization,
            order_size,lr_encode)
            for _ in range(n_layers-sub_layers)
        ))

    def forward(self, x, mask=None):

        assert mask is None, "TODO mask not yet supported!"

        # Batch multiply to get initial embeddings of nodes
        h = self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1) if self.init_embed is not None else x

        graph_size = h.size()[1]

        h1 = h[:,:graph_size-self.order_size]
        h2 = h[:,graph_size-self.order_size:]

        h1 = self.layers1(h1)
        h2 = self.layers2(h2)
           
        h = torch.cat((h1, h2), dim=1)

        h = self.layers(h)

        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )

```

2. class MultiHeadAttentionLayer
```py
class MultiHeadAttentionLayer(nn.Sequential):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden=512,
            normalization='batch',
            order_size=0,
            lr_encode=1.0
    ):
        super(MultiHeadAttentionLayer, self).__init__(
            SkipConnection(
                MultiHeadAttention(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim,
                    order_size=order_size,
                    lr_encode=lr_encode
                )
            ),
            Normalization(embed_dim, normalization),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim, normalization)
        )
```

3. class MultiHeadAttention
```py
class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim,
            order_size=0,
            lr_encode=1.0,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.order_size = order_size
        self.lr_encode = lr_encode

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        self.W_out = nn.Parameter(torch.Tensor(n_heads, val_dim, embed_dim))
        
        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """

        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size ,key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)
        
        w_l_r = torch.ones(graph_size)  
        if self.order_size > 0 and self.lr_encode != 1.0 :
          w_l_r = torch.tensor([1]*(graph_size-self.order_size) + [self.lr_encode**i for i in range(self.order_size)])
        w_r = torch.ones(graph_size,graph_size) 
        if self.order_size > 0 and self.lr_encode != 1.0 :
           w_r = torch.matmul(w_l_r.unsqueeze(1),w_l_r.unsqueeze(0))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        W_r = w_r.unsqueeze(0).unsqueeze(0).repeat(self.n_heads, batch_size, 1, 1).to(device)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))
        compatibility = torch.matmul(compatibility,W_r)

        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -np.inf

        attn = torch.softmax(compatibility, dim=-1)

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc

        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        # Alternative:
        # headst = heads.transpose(0, 1)  # swap the dimensions for batch and heads to align it for the matmul
        # # proj_h = torch.einsum('bhni,hij->bhnj', headst, self.W_out)
        # projected_heads = torch.matmul(headst, self.W_out)
        # out = torch.sum(projected_heads, dim=1)  # sum across heads

        # Or:
        # out = torch.einsum('hbni,hij->bnj', heads, self.W_out)

        return out
```

## 5. 对优先限制 mask 的更改
###### AttentionModel/_inner()
```py
        state = self.problem.make_state(input)
        state.add_order(self.order_size)
```

###  problems/tsp/state_tsp.py
1. constrain
```py
    constrain_: torch.Tensor # Keeps track of nodes with constrain (batch,1,n_loc+1)
    
    @property
    def constrain(self):
        if self.constrain_.dtype == torch.uint8:
            return self.constrain_[:,:,:-1]
        else:
            return mask_long2bool(self.constrain_[:,:,:-1], n=self.loc.size(-2))
```

2. add_order(order_size)
```py
    def add_order(self, order_size):
        
        if order_size ==0 :
            return self

        batch, graph, _ = self.loc.size()
        order_ = range(0,graph-order_size+1)
        order = []
        for i in order_:
         order += [torch.ones(batch)*(i) ]  
        constrain_ = self.constrain_
        device=self.constrain_.device
        if self.constrain_.dtype == torch.uint8:
            for selected in order:
              node = selected[:, None].to(device).to(torch.int64)
              constrain_ = constrain_.scatter(-1, node[:, :, None], 1)
        else:
            for selected in order:
              node = selected[:, None].to(device).to(torch.int64)
              constrain_ = mask_long_scatter(constrain_, node)

        return self._replace(constrain_=constrain_)
```

3. update(selected)
```py
    def update(self, selected):

        # Update the state
        prev_a = selected[:, None]  # Add dimension for step

        # Add the length
        # cur_coord = self.loc.gather(
        #     1,
        #     selected[:, None, None].expand(selected.size(0), 1, self.loc.size(-1))
        # )[:, 0, :]
        cur_coord = self.loc[self.ids, prev_a]
        lengths = self.lengths
        if self.cur_coord is not None:  # Don't add length for first action (selection of start node)
            lengths = self.lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)  # (batch_dim, 1)

        # Update should only be called with just 1 parallel step, in which case we can check this way if we should update
        first_a = prev_a if self.i.item() == 0 else self.first_a

        if self.visited_.dtype == torch.uint8:
            # Add one dimension since we write a single value
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        else:
            visited_ = mask_long_scatter(self.visited_, prev_a)

        refree = selected + 1 
        refree = refree[:,None]
        if self.contrain_.dtype == torch.uint8:
            # Add one dimension since we write a single value
            contrain_ = self.contrain_.scatter(-1, refree[:, :, None], 1)
        else:
            contrain_ = mask_long_scatter(self.contrain_, refree)

        return self._replace(first_a=first_a, prev_a=prev_a, visited_=visited_,
                             lengths=lengths, cur_coord=cur_coord, i=self.i + 1,
                             contrain_=contrain_)
```

4. get_mask()
```py
    def get_mask(self):
        return (self.visited > 0) | (self.constrain < 1)   # Hacky way to return bool or uint8 depending on pytorch version
```

