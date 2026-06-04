import torch
import torch.nn as nn
# from thop import profile 

class MultiQueryAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.0):
        """
        Multi-Query Attention 的实现。
        Args:
            hidden_size (int): 输入特征的维度，也即 hidden_state 的最后一维。
            num_heads (int): 注意力头的数量。
            dropout (float): dropout 的概率，默认为 0.0。
        """
        super(MultiQueryAttention, self).__init__()

        assert hidden_size % num_heads == 0, "hidden_size 必须能被 num_heads 整除"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads  # 每个头的维度

        # 定义线性变换层，用于生成 Q, K, V
        self.query = nn.Linear(hidden_size, hidden_size)  # 每个头独立的 Query
        self.key = nn.Linear(hidden_size, self.head_dim)  # 所有头共享的 Key
        self.value = nn.Linear(hidden_size, self.head_dim)  # 所有头共享的 Value

        self.dropout = nn.Dropout(dropout)
        self.out_projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_state, attention_mask=None):
        """
        前向传播函数。
        Args:
            hidden_state (torch.Tensor): 输入的 hidden_state，形状为 [batch_size, seq_len, hidden_size]。
            attention_mask (torch.Tensor, optional): 注意力掩码，用于屏蔽某些位置，形状为 [batch_size, seq_len]。默认为 None。
        Returns:
            torch.Tensor: 注意力输出，形状为 [batch_size, seq_len, hidden_size]。
        """
        batch_size, seq_len, _ = hidden_state.size()

        # 1. 通过线性层得到 Q, K, V
        query = self.query(hidden_state)  # [batch_size, seq_len, hidden_size]
        key = self.key(hidden_state)      # [batch_size, seq_len, head_dim]
        value = self.value(hidden_state)  # [batch_size, seq_len, head_dim]

        # 2. 将 Q 拆分为多头
        # [batch_size, num_heads, seq_len, head_dim]
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  

        # 3. 扩展 K 和 V 到 num_heads 维度（所有头共享相同的 K/V）
        # unsqueeze(1)把第一维插进去[batch_size, 1, seq_len, head_dim]
        # 第 1 个 -1：这一维保持原样，也就是 batch_size 不变
        # 第 2 个 self.num_heads：把这一维从 1 扩成 num_heads
        # 第 3 个 -1：这一维保持原样，也就是 seq_len 不变
        # 第 4 个 -1：这一维保持原样，也就是 head_dim 不变
        key = key.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # [batch_size, num_heads, seq_len, head_dim]
        value = value.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # [batch_size, num_heads, seq_len, head_dim]

        # 4. 计算注意力权重
        attention_weights = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [batch_size, num_heads, seq_len, seq_len]
        # 应用 attention mask
        if attention_mask is not None:
            attention_weights = attention_weights.masked_fill(attention_mask[:, None, None, :] == 0, float('-inf'))

        attention_weights = torch.softmax(attention_weights, dim=-1)  # [batch_size, num_heads, seq_len, seq_len]
        attention_weights = self.dropout(attention_weights)

        # 5. 计算上下文向量
        context = torch.matmul(attention_weights, value)  # [batch_size, num_heads, seq_len, head_dim]

        # 6. 将多头合并
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)  # [batch_size, seq_len, hidden_size]

        # 7. 通过输出线性层
        output = self.out_projection(context)  # [batch_size, seq_len, hidden_size]

        return output

if __name__ == '__main__':
    # 示例
    batch_size = 2
    seq_len = 10
    hidden_size = 256
    num_heads = 8

    # 创建一个 MQA 实例
    mqa = MultiQueryAttention(hidden_size, num_heads)

    # 创建一个随机的 hidden_state
    hidden_state = torch.randn(batch_size, seq_len, hidden_size)

    # 创建一个 attention mask (可选)
    attention_mask = torch.ones(batch_size, seq_len)
    attention_mask[:, 5:] = 0  # 屏蔽掉每个 batch 中 seq_len 的后 5 个位置

    # 通过 MQA 层
    output = mqa(hidden_state, attention_mask)

    # 打印输出形状
    print("输出形状:", output.shape)  # torch.Size([2, 10, 256])