import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CustomMHA(nn.Module):
    def __init__(self, num_heads, d_model, dropout = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model//num_heads
        self.scale = 1.0/math.sqrt(self.d_k)

        self.Win = nn.Linear(d_model, 3*num_heads*self.d_k)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask = False): #x.shape = [B, L, d_model]
        B, L, _ = x.shape

        x_qkv = self.Win(x)
        Q, K, V = x_qkv.split(self.d_model, dim = -1)

        Q = Q.view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.d_k).transpose(1, 2)

        S = (Q @ K.transpose(-2 ,-1)) * self.scale #B, h, L, L

        if causal_mask:
            mask = torch.triu(torch.ones(L, L, device = x.device), diagonal = 1).bool()
            mask = mask.expand(B, self.num_heads, -1, -1)
            S = S.masked_fill(mask, float('-inf'))

        A = torch.softmax(S, dim = -1)
        A = self.dropout(A)

        z = A @ V
        z = z.transpose(1, 2).reshape(B, L, self.d_model)

        return self.proj(z)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_MLP, dropout=0.1):
        super().__init__()
        self.attention = CustomMHA(num_heads=num_heads, d_model=d_model, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.MLP = nn.Sequential(
            nn.Linear(d_model, dim_MLP),
            nn.GELU(),
            nn.Linear(dim_MLP, d_model)
        )
    
    def forward(self, x):
        y = self.norm1(x)
        y = self.attention(y)
        x = x + y

        y = self.norm2(x)
        y = self.MLP(y)
        x = x + y

        return x

class EncoderTransformer(nn.Module): #timestep embeddings assumed to be handled by user
    def __init__(self, num_classes, d_model, num_heads, num_encoder_layers = 4, dim_MLP = 1024, use_cls_token = True):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model = d_model, num_heads = num_heads, dim_MLP = dim_MLP)
            for _ in range(num_encoder_layers)
        ])
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pool = nn.AdaptiveAvgPool1d(output_size = 1)
        self.classifier = nn.Linear(d_model, num_classes)
        self.use_cls_token = use_cls_token
    
    def forward(self, x): #x.shape = [B, L, d_model]
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim = 1)

        for layer in self.layers:
            x = layer(x)

        if self.use_cls_token:
            return self.classifier(x[:, 0, :])

        else:
            x = x.transpose(1, 2) #[B, d_model, L] is the expected shape for the pooling layer, which pools L -> 1
            x = self.pool(x).squeeze(-1)
            return self.classifier(x) #returns logits