import torch
import torch.nn as nn

class GroupedQueryAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, group_size=2, dropout=0.0):
        """
        Grouped Query Attention 实现。
        Args:
            hidden_size (int): 输入特征的维度。
            num_heads (int): 查询头的数量。
            group_size (int): 每个组中包含的查询头数量。
            dropout (float): dropout 的概率。
        """
        super(GroupedQueryAttention, self).__init__()
        assert hidden_size % num_heads == 0, "hidden_size 必须能被 num_heads 整除"
        assert num_heads % group_size == 0, "num_heads 必须能被 group_size 整除"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.group_size = group_size
        self.group_num = num_heads // group_size
        self.head_dim = hidden_size // num_heads 

        # 查询头
        self.query = nn.Linear(hidden_size, hidden_size)
        # 键和值头（分组共享）
        self.key = nn.Linear(hidden_size, self.group_num * self.head_dim)
        self.value = nn.Linear(hidden_size, self.group_num * self.head_dim)

        self.dropout = nn.Dropout(dropout)
        self.out_projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_state, attention_mask=None):
        """
        前向传播函数。
        Args:
            hidden_state (torch.Tensor): 输入张量，形状为 [batch_size, seq_len, hidden_size]。
            attention_mask (torch.Tensor, optional): 注意力掩码，形状为 [batch_size, seq_len]。
        Returns:
            torch.Tensor: 注意力输出，形状为 [batch_size, seq_len, hidden_size]。
        """
        batch_size, seq_len, _ = hidden_state.size()

        # 1. 通过线性层生成 Q, K, V
        query = self.query(hidden_state)  # [batch_size, seq_len, hidden_size]
        key = self.key(hidden_state)      # [batch_size, seq_len, group_num * head_dim]
        value = self.value(hidden_state)  # [batch_size, seq_len, group_num * head_dim]

        # 2. 将 Q, K, V 拆分成多头
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]

        # K 和 V 扩展到 num_heads 个头
        key = key.view(batch_size, seq_len, self.group_num, self.head_dim).transpose(1, 2)  # [batch_size, group_num, seq_len, head_dim]
        # unsqueeze(2)把第2维插进去 [batch_size, group_num, 1, seq_len, head_dim]
        # expand(-1, -1, self.group_size, -1, -1) 把第2维从1扩成group_size [batch_size, group_num, group_size, seq_len, head_dim]
        key = key.unsqueeze(2).expand(-1, -1, self.group_size, -1, -1).contiguous().view(batch_size, -1, seq_len, self.head_dim)  # [batch_size, num_heads, seq_len, head_dim]
        value = value.view(batch_size, seq_len, self.group_num, self.head_dim).transpose(1, 2)  # [batch_size, group_num, seq_len, head_dim]
        value = value.unsqueeze(2).expand(-1, -1, self.group_size, -1, -1).contiguous().view(batch_size, -1, seq_len, self.head_dim)  # [batch_size, num_heads, seq_len, head_dim]

        # 3. 计算注意力权重
        attention_weights = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if attention_mask is not None:
            attention_weights = attention_weights.masked_fill(attention_mask[:, None, None, :] == 0, float('-inf'))
        attention_weights = torch.softmax(attention_weights, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 4. 计算上下文向量
        context = torch.matmul(attention_weights, value)

        # 5. 合并多头
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        # 6. 输出投影
        output = self.out_projection(context)
        return output

# 示例
if __name__ == '__main__':
    batch_size = 2
    seq_len = 10
    hidden_size = 256
    num_heads = 8
    group_size = 2  # 每组 2 个头，共 4 组

    gqa = GroupedQueryAttention(hidden_size, num_heads, group_size)
    hidden_state = torch.randn(batch_size, seq_len, hidden_size)
    attention_mask = torch.ones(batch_size, seq_len)

    attention_mask[:, 5:] = 0  # 屏蔽后 5 个位置
    output = gqa(hidden_state, attention_mask)
    print("输出形状:", output.shape)  # torch.Size([2, 10, 256])