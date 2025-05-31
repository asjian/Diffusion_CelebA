import torch
import torch.nn as nn
from transformer import CustomMHA

class SelfAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout = 0.1):
        super().__init__()

        self.norm = nn.LayerNorm(normalized_shape = d_model)
        self.mha = nn.MultiheadAttention(embed_dim = d_model, num_heads = num_heads, dropout = dropout)
        self.custommha = CustomMHA(num_heads = 8, d_model = d_model, dropout = dropout)
        self.dropout = nn.Dropout(p = dropout)
    
    def forward(self, x):
        B, C, H, W = x.shape
        #x_seq = x.view(B, C, H*W).permute(2, 0, 1) #this ordering is for torch MHA
        x_seq = x.view(B, C, H*W).transpose(1, 2)
        x_norm = self.norm(x_seq)

        #attn_output, _ = self.mha(query = x_norm, key = x_norm, value = x_norm, need_weights = False)
        attn_output = self.custommha(x_norm)
        #attn_output = self.dropout(attn_output) #this is optional since there's already dropout in the torch MHA

        attn_output = attn_output + x_seq
        #out = attn_output.permute(1, 2, 0).view(B, C, H, W) #this ordering is also for torch MHA
        out = attn_output.transpose(1, 2).reshape(B, C, H, W)

        return out