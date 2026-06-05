import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardAttention(nn.Module):
    def __init__(self, hidden_size, dropout=0.0):
        """
        标准单头注意力机制的教学版实现。
        Args:
            hidden_size (int): 输入特征的维度，也即 hidden_state 的最后一维。
            dropout (float): dropout 的概率，默认为 0.0。
        """
        super(StandardAttention, self).__init__()

        self.hidden_size = hidden_size

        # 定义线性变换层，用于生成 Q, K, V
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.out_projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_state, attention_mask=None):
        """
        前向传播函数。
        Args:
            hidden_state (torch.Tensor): 输入的 hidden_state，形状为 [batch_size, seq_len, hidden_size]。
            attention_mask (torch.Tensor, optional): 注意力掩码，用于屏蔽某些位置，形状为 [batch_size, seq_len]。
        Returns:
            torch.Tensor: 注意力输出，形状为 [batch_size, seq_len, hidden_size]。
        """
        batch_size, seq_len, _ = hidden_state.size()

        # 1. 通过线性层得到 Q, K, V
        query = self.query(hidden_state)  # [batch_size, seq_len, hidden_size]
        key = self.key(hidden_state)      # [batch_size, seq_len, hidden_size]
        value = self.value(hidden_state)  # [batch_size, seq_len, hidden_size]

        # 2. 计算注意力权重。标准 attention 会显式构造 [seq_len, seq_len] 的注意力矩阵
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.hidden_size ** 0.5)  # [batch_size, seq_len, seq_len]
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask[:, None, :] == 0, float('-inf'))

        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 3. 用注意力权重对 V 加权求和
        context = torch.matmul(attention_weights, value)  # [batch_size, seq_len, hidden_size]

        # 4. 输出投影
        output = self.out_projection(context)
        return output


class LinearAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.0, eps=1e-6):
        """
        Linear Attention 的教学版实现。

        核心想法：
        标准 attention 是 softmax(QK^T)V，会先得到 [seq_len, seq_len] 的矩阵。
        Linear Attention 使用非负特征映射 phi，把计算近似改写为：
            phi(Q) @ (phi(K)^T @ V)
        这样可以先计算 phi(K)^T @ V，避免显式构造 seq_len x seq_len 矩阵。

        Args:
            hidden_size (int): 输入特征的维度，也即 hidden_state 的最后一维。
            num_heads (int): 注意力头的数量。
            dropout (float): dropout 的概率，默认为 0.0。
            eps (float): 防止归一化分母为 0 的小常数。
        """
        super(LinearAttention, self).__init__()

        assert hidden_size % num_heads == 0, "hidden_size 必须能被 num_heads 整除"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.eps = eps

        # 定义线性变换层，用于生成 Q, K, V
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.out_projection = nn.Linear(hidden_size, hidden_size)

    def feature_map(self, x):
        """
        非负特征映射 phi。
        elu(x) + 1 简单稳定，常用于 Linear Attention 的教学实现。
        """
        return F.elu(x) + 1

    def forward(self, hidden_state):
        """
        前向传播函数。
        Args:
            hidden_state (torch.Tensor): 输入的 hidden_state，形状为 [batch_size, seq_len, hidden_size]。
            attention_mask (torch.Tensor, optional): 注意力掩码，用于屏蔽某些 key/value 位置，形状为 [batch_size, seq_len]。
        Returns:
            torch.Tensor: 注意力输出，形状为 [batch_size, seq_len, hidden_size]。
        """
        batch_size, seq_len, _ = hidden_state.size()

        # 1. 通过线性层得到 Q, K, V
        query = self.query(hidden_state)  # [batch_size, seq_len, hidden_size]
        key = self.key(hidden_state)      # [batch_size, seq_len, hidden_size]
        value = self.value(hidden_state)  # [batch_size, seq_len, hidden_size]

        # 2. 将 Q, K, V 拆分成多头
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)      # [batch_size, num_heads, seq_len, head_dim]
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]

        # 3. 对 Q 和 K 做非负特征映射
        query = self.feature_map(query)  # [batch_size, num_heads, seq_len, head_dim]
        key = self.feature_map(key)      # [batch_size, num_heads, seq_len, head_dim]

        # 4. 应用 attention mask。Linear Attention 不显式构造注意力矩阵，
        #    所以这里直接把被屏蔽位置的 K/V 置零，使它们不参与后续聚合。
        # if attention_mask is not None:
        #     mask = attention_mask[:, None, :, None]  # [batch_size, 1, seq_len, 1]
        #     key = key * mask
        #     value = value * mask

        value = self.dropout(value)

        # 5. 先计算 K^T V，把序列长度维度聚合掉
        kv = torch.matmul(key.transpose(-2, -1), value)  # [batch_size, num_heads, head_dim, head_dim]

        # 6. 再用 Q 读取聚合后的上下文
        context = torch.matmul(query, kv)  # [batch_size, num_heads, seq_len, head_dim]

        # 7. 归一化。相当于除以每个 query 对所有 key 的权重和
        key_sum = key.sum(dim=-2)  # [batch_size, num_heads, head_dim]
        denominator = torch.matmul(query, key_sum.unsqueeze(-1)).clamp_min(self.eps)  # [batch_size, num_heads, seq_len, 1]
        context = context / denominator

        # 8. 合并多头
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        # 9. 输出投影
        output = self.out_projection(context)
        return output


if __name__ == '__main__':
    # 示例
    batch_size = 2
    seq_len = 10
    hidden_size = 256
    num_heads = 8

    # 创建标准单头 Attention 和 Linear Attention 实例
    standard_attention = StandardAttention(hidden_size)
    linear_attention = LinearAttention(hidden_size, num_heads)

    # 创建一个随机的 hidden_state
    hidden_state = torch.randn(batch_size, seq_len, hidden_size)

    # 创建一个 attention mask (可选)
    # attention_mask = torch.ones(batch_size, seq_len)
    # attention_mask[:, 5:] = 0  # 屏蔽掉每个 batch 中 seq_len 的后 5 个位置

    # 通过 Attention 层
    standard_output = standard_attention(hidden_state)
    linear_output = linear_attention(hidden_state)

    # 打印输出形状
    print("标准 Attention 输出形状:", standard_output.shape)  # torch.Size([2, 10, 256])
    print("Linear Attention 输出形状:", linear_output.shape)  # torch.Size([2, 10, 256])
