import torch
import torch.nn as nn
import numpy as np
import pickle
from config import args
# ====================================================================================================
# Transformer模型
# 定义Embeddings类来实现文本嵌入层，这里s说明代表两个一模一样的嵌入层, 他们共享参数.
# 该类继承nn.Module, 这样就有标准层的一些功能, 这里我们也可以理解为一种模式, 我们自己实现的所有层都会这样去写.
device = torch.device('cuda:{}'.format(args.gpu))
class Embedding(nn.Module):
    def __init__(self, vocab_size, max_len, d_model):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)  #

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long).to(device)
        pos = pos.unsqueeze(0).expand_as(x).to(device)
        embedding = self.tok_embed(x) + self.pos_embed(pos)
        return self.norm(embedding)

def get_attn_pad_mask(seq_q, seq_k):
    # pad mask的作用：在对value向量加权平均的时候，可以让pad对应的alpha_ij=0，这样注意力就不会考虑到pad向量
    """这里的q,k表示的是两个序列（跟注意力机制的q,k没有关系），例如encoder_inputs (x1,x2,..xm)和encoder_inputs (x1,x2..xm)
    encoder和decoder都可能调用这个函数，所以seq_len视情况而定
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    """
    batch_size, len_q = seq_q.size()  # 这个seq_q只是用来expand维度的
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    # 例如:seq_k = [[1,2,3,4,0], [1,2,3,5,0]]
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k] 构成一个立方体(batch_size个这样的矩阵)

# ==========================================================================================
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask, d_k=64):
        """
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        说明：在encoder-decoder的Attention层中len_q(q1,..qt)和len_k(k1,...km)可能不同
        """
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        # mask矩阵填充scores（用-1e9填充scores中与attn_mask中值为1位置相对应的元素）
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)  # 对最后一个维度(v)做softmax
        # scores : [batch_size, n_heads, len_q, len_k] * V: [batch_size, n_heads, len_v(=len_k), d_v]
        context = torch.matmul(attn, V)  # context: [batch_size, n_heads, len_q, d_v]
        # context：[[z1,z2,...],[...]]向量, attn注意力稀疏矩阵（用于可视化的）
        return context, attn


## 6. MultiHeadAttention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=64, d_v=64, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        ## 输入进来的QKV是相等的，我们会使用映射linear做一个映射得到参数矩阵Wq, Wk,Wv
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        torch.nn.init.xavier_uniform_(self.W_Q.weight)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        torch.nn.init.xavier_uniform_(self.W_K.weight)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        torch.nn.init.xavier_uniform_(self.W_V.weight)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        torch.nn.init.xavier_uniform_(self.linear.weight)
        self.layer_norm = nn.LayerNorm(d_model)
        # self.dropout = nn.Dropout(dropout)
    def forward(self, Q, K, V, attn_mask, n_heads, d_k=64, d_v=64):

        ## 这个多头分为这几个步骤，首先映射分头，然后计算atten_scores，然后计算atten_value;
        ##输入进来的数据形状： Q: [batch_size x len_q x d_model], K: [batch_size x len_k x d_model], V: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)

        ##下面这个就是先映射，后分头；一定要注意的是q和k分头之后维度是一致额，所以一看这里都是dk
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

        ## 输入进行的attn_mask形状是 batch_size x len_q x len_k，然后经过下面这个代码得到 新的attn_mask : [batch_size x n_heads x len_q x len_k]，就是把pad信息重复了n个头上
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)


        ##然后我们计算 ScaledDotProductAttention 这个函数，去7.看一下
        ## 得到的结果有两个：context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q x len_k]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size x len_q x n_heads * d_v]
        output = self.linear(context)
        return self.layer_norm(output + residual), attn # output: [batch_size x len_q x d_model]


# ## 8. PoswiseFeedForwardNet
# class PoswiseFeedForwardNet(nn.Module):
#     def __init__(self, d_model, d_ff, dropout=0.1):
#         super(PoswiseFeedForwardNet, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
#         self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
#         self.layer_norm = nn.LayerNorm(d_model)
#         # self.dropout = nn.Dropout(dropout)
#     def forward(self, inputs):
#         residual = inputs # inputs : [batch_size, len_q, d_model]
#         output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
#         output = self.conv2(output).transpose(1, 2)
#         return self.layer_norm(output + residual)

## 8. PoswiseFeedForwardNet
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        torch.nn.init.kaiming_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.layer_norm = nn.LayerNorm(d_model)
    def forward(self, inputs):
        residual = inputs  # inputs : [batch_size, len_q, d_model]
        output = nn.ReLU()(self.fc1(inputs))
        output = self.fc2(output)
        return self.layer_norm(output + residual)



class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)
        self.n_heads = n_heads
    def forward(self, enc_inputs, enc_self_attn_mask):
        """E
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]  mask矩阵(pad mask or sequence mask)
        """
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        # 第一个enc_inputs * W_Q = Q
        # 第二个enc_inputs * W_K = K
        # 第三个enc_inputs * W_V = V
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                               enc_self_attn_mask, self.n_heads)  # enc_inputs to same Q,K,V（未线性变换前）
        enc_outputs = self.pos_ffn(enc_outputs)
        # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self, eventlog, d_model, n_heads, n_layers, d_ff, f):
        super(Encoder, self).__init__()
        dict_cat_view = {}
        dict_cat_emb = {}
        # 读取该日志的基本属性:数值属性列表,类别属性列表,所有轨迹的平均长度
        with open("data/" + eventlog + "/" + eventlog + '_num_cols.pickle', 'rb') as pickle_file:
            self.num_view = pickle.load(pickle_file)
        with open("data/" + eventlog + "/" + eventlog + '_cat_cols.pickle', 'rb') as pickle_file:
            self.cat_view = pickle.load(pickle_file)
        with open("data/" + eventlog + "/" + eventlog + '_seq_length.pickle', 'rb') as pickle_file:
            self.seq_length = pickle.load(pickle_file)
        for c in self.cat_view:
            voca_size = np.load("data/" + eventlog + "/" + eventlog + '_' + c + '_' + str(f) + "_info.npy")  # 词表数目(还不包括填充0)
            if c == "activity":
                dict_cat_view[c] = [voca_size + 2, d_model]
                dict_cat_emb[c] = Embedding(voca_size + 2, self.seq_length+1, d_model).to(device)
            else:
                dict_cat_view[c] = [voca_size + 1, d_model]
                dict_cat_emb[c] = Embedding(voca_size + 1, self.seq_length, d_model).to(device)
        self.dict_cat_view = dict_cat_view
        self.dict_cat_emb = dict_cat_emb
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)])
    def forward(self, att_str, att):
        """
        enc_inputs: [batch_size, src_len]
        """
        att_dict = dict(zip(att_str, att))
        # print(pr.format(i, *att))
        # 嵌入层
        list_cat_view = []
        list_cat_view_original = []
        for c in self.cat_view:
            embedding = self.dict_cat_emb.get(c)
            enc_outputs = embedding(att_dict.get(c))  # [batch_size, src_len, d_model]
            list_cat_view.append(enc_outputs)
            list_cat_view_original.append(att_dict.get(c))
        enc_outputs = torch.cat(list_cat_view, 1)
        outputs_original = torch.cat(list_cat_view_original, 1)
        # Encoder输入序列的pad mask矩阵
        enc_self_attn_mask = get_attn_pad_mask(outputs_original, outputs_original)  # [batch_size, src_len, src_len]
        enc_self_attns = []  # 在计算中不需要用到，它主要用来保存你接下来返回的attention的值（这个主要是为了你画热力图等，用来看各个词之间的关系
        for layer in self.layers:  # for循环访问nn.ModuleList对象
            # 上一个block的输出enc_outputs作为当前block的输入
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs,
                                               enc_self_attn_mask)  # 传入的enc_outputs其实是input，传入mask矩阵是因为你要做self attention
            enc_self_attns.append(enc_self_attn)  # 这个只是为了可视化
        return enc_outputs, enc_self_attns


class Transformer(nn.Module):
    def __init__(self, eventlog, d_model, n_heads, n_layers, d_ff, f):
        super(Transformer, self).__init__()

        self.activity_vocab_size = np.load("data/" + eventlog + "/" + eventlog + '_' + 'activity' + '_' + str(f) + "_info.npy")  # 词表数目(还不包括填充0)

        self.encoder = Encoder(eventlog, d_model, n_heads, n_layers, d_ff, f)
        self.fc = nn.Linear(d_model, d_model)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        self.active = nn.Tanh()
        self.projection = nn.Linear(d_model, self.activity_vocab_size, bias=False)
        torch.nn.init.xavier_uniform_(self.projection.weight)

    def forward(self, att_str, att):
        """Transformers的输入：两个序列
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        """
        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        # 经过Encoder网络后，得到的输出还是[batch_size, src_len, d_model]
        enc_outputs, enc_self_attns = self.encoder(att_str, att)
        # dec_outputs: [batch_size, tgt_len, d_model] -> dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        # 后续将句子所在的维度做pooling，所以将句子所在维度放到最后面
        # enc_outputs = enc_outputs.transpose(2, 1)  # seq_len所在维度由原先的第二维变成第三维[batch_size, embed_dim, seq_len]
        # # [batch_size, embed_dim, 1]
        # enc_outputs = F.avg_pool1d(enc_outputs, kernel_size=seq_length)
        # [batch_size, embed_dim]
        # enc_outputs = enc_outputs.squeeze(-1)
        enc_outputs = self.active(self.fc(enc_outputs[:, 0]))
        dec_logits = self.projection(enc_outputs)
        return dec_logits, enc_self_attns