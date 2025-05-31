import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.d = emb_dim

        self.MLP = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )

        self.linear = nn.Linear(emb_dim, emb_dim)

    def sin_embedding(self, t):
        d = self.d
        t = torch.reshape(t, (t.shape[0], 1))
        t = t.float()

        half_dim = d//2
        indices = torch.arange(0, half_dim, device = t.device, dtype = torch.float)
        powers = 2 * indices/d

        base = torch.tensor(10000.0, device = t.device)
        exps = torch.pow(base, powers)
        exps = torch.reshape(exps, (1, exps.shape[0]))

        fracs = t/exps
        sins = torch.sin(fracs)
        cosines = torch.cos(fracs)

        emb = torch.stack((sins, cosines), dim = -1)
        emb = emb.view(emb.shape[0], d)

        return emb
    
    def forward(self, t):
        sin_emb = self.sin_embedding(t)
        #return self.MLP(sin_emb)
        return self.linear(sin_emb)
        
