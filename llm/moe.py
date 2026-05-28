# 主要参考自 mistral MOE 的实现
from logging import config

from logging import config

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicExpert(nn.Module):
    def __init__(self, feature_in, feature_out):
        super().__init__()
        self.fc = nn.Linear(feature_in, feature_out)

    def forward(self, x):
        return self.fc(x)

class MOERouter(nn.Module):
    def __init__(self, hidden_dim, expert_number, top_k):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, expert_number)
        self.expert_number = expert_number
        self.top_k = top_k
    
    def forward(self, hidden_states):
        # 计算路由logits [8*4]
        router_logits = self.gate(hidden_states)  # shape is (b * s, expert_number) 
        
        # 计算专家经过softmax之后的概率 [8*4]
        routing_probs = F.softmax(router_logits, dim=-1, dtype=torch.float)
        
        # 计算topk的专家的输出 [8*2] 每个token的topk权重和索引
        router_weights, selected_experts = torch.topk(
            routing_probs, self.top_k, dim=-1
        )  # shape都是 (b * s, top_k) 
        
        # 专家权重归一化，表示沿着最后一个维度求和，(b*s,) 变成 (b*s, 1)
        router_weights = router_weights / router_weights.sum(dim=-1, keepdim=True)
        router_weights = router_weights.to(hidden_states.dtype)
        
        # 生成专家掩码 [8,2,4] 每个token的topk专家的one-hot编码
        expert_mask = F.one_hot(
            selected_experts,
            num_classes=self.expert_number
        )  # shape是 (b * s, top_k, expert_number)
        expert_mask = expert_mask.permute(2, 1, 0)  # (expert_number, top_k, b * s)
        
        return router_logits, router_weights, selected_experts, expert_mask



class SparseMOE(nn.Module):
    def __init__(self, hidden_dim, expert_number, top_k):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.expert_number = expert_number
        self.top_k = top_k

        self.experts = nn.ModuleList(
            [
                BasicExpert(self.hidden_dim, self.hidden_dim) for _ in range(self.expert_number)
            ]
        )

        self.router = MOERouter(self.hidden_dim, self.expert_number, self.top_k)
    
    def forward(self, x):
        # x shape is (b, s, hidden_dim)
        batch_size, seq_len, hidden_dim = x.size()

        # 合并前两个维度，因为按照token进行路由
        # [8*16]
        hidden_states = x.view(-1, hidden_dim) # shape is(b * s, hidden_dim)

        # [8*4, 8*2, 8*2, 4*2*8] 
        router_logits, router_weights, selected_experts_indices, expert_mask = self.router(hidden_states)
        # 其中 selected_experts_indices shape 是 (b * s, top_k)
        # 其中 expert_mask shape 是 (expert_number, top_k, b * s)
        
        final_hidden_states = torch.zeros(
            (batch_size * seq_len, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device
        ) # shape is (b * s, hidden_dim)
				
		
        for expert_idx in range(self.expert_number):
            expert_layer = self.experts[expert_idx]
            # expert_mask[expert_idx] 
            # [[1, 1, 1, 1, 0, 1, 1, 1],
            #  [0, 0, 0, 0, 1, 0, 0, 0]]
            # 第 0 行：这个 expert 是哪些 token 的 top1
            # 第 1 行：这个 expert 是哪些 token 的 top2
            idx, top_x = torch.where(expert_mask[expert_idx]) 

            # 从所有 token 的 hidden_states 里，取出“当前这个 expert 需要处理的那些 token”。
            # hidden_states 的 shape 是 (b * s, hidden_dim)
            # hidden_states.unsqueeze(0) (b*s, hidden_dim) -> (1, b*s, hidden_dim)
            # [:, top_x, :]在第二维上按 top_x 取索引：取出当前 expert 负责的那些 token
            # .reshape(-1, hidden_dim) ：变成(selected_token_number, hidden_dim)
            current_state = hidden_states.unsqueeze(0)[:, top_x, :].reshape(-1, hidden_dim) # （selected_token_number, hidden_dim）

            # router_weight 的 shape 是 (b * s, top_k)
            # current_hidden_states = expert_layer(
            #     current_state
            # ) * router_weights[top_x, idx].unsqueeze(-1)  # （selected_token_number, 1） 这里有广播

            expert_out = expert_layer(current_state) # （selected_token_number, hidden_dim）= 5*16
            # （selected_token_number, 1），增加一维，相当于矩阵乘法变成标量乘法
            select_weights = router_weights[top_x, idx].unsqueeze(-1) 
            current_hidden_states = expert_out * select_weights

            # 把当前专家的输出加到 final_hidden_states 中
            # current_hidden_states [selected_token_number, hidden_dim]
            # final_hidden_states [b * s, hidden_dim] 这是所有 token 的总输出缓冲区
            final_hidden_states[top_x] += current_hidden_states.to(hidden_states.dtype)

        # [b * s, hidden_dim] -> [b, s, hidden_dim]
        final_hidden_states = final_hidden_states.reshape(batch_size, seq_len, hidden_dim)

        return final_hidden_states, router_logits # shape 是 (b * s, expert_number)


def test_token_level_moe():
    x = torch.rand(2, 4, 16) # bs, seq_len, hidden_dim
    token_level_moe = SparseMOE(16, 4, 2) # hidden_dim, expert_number, top_k
    out = token_level_moe(x)
    print(out[0].shape, out[1].shape) # (2, 4, 16) (8, 4)


test_token_level_moe()