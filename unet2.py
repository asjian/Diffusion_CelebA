import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from sinusoidal_embedding import Embedding

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim, num_groups = 32, dropout = 0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_dim = emb_dim

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)
        self.gn1 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups, out_channels)
        
        self.time_emb_proj = nn.Linear(emb_dim, out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x, t_emb = None):
        h = self.activation(self.gn1(self.conv1(x)))
        if t_emb is not None:
            t_proj = self.time_emb_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
            h = h + t_proj

        h = self.activation(self.gn2(self.conv2(h)))
        h = self.dropout(h)
        
        return self.residual_conv(x) + h
        
class UNet2(nn.Module):
    def __init__(self, in_channels, base_channels = 64, channel_mults = [1,2,4,8], emb_dim = 256):
        super().__init__()
        self.in_channels = in_channels
        self.emb_dim = emb_dim
        self.emb = Embedding(emb_dim)
        self.activation = nn.GELU()

        self.down_layers = nn.ModuleList()
        self.first_conv = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        
        curr_channels = base_channels
        for mult in channel_mults:
            out_channels = base_channels * mult
            self.down_layers.append(ResidualBlock(curr_channels, out_channels, emb_dim))
            self.down_layers.append(ResidualBlock(out_channels, out_channels, emb_dim))
            self.down_layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1))
            curr_channels = out_channels
        
        self.bottleneck1 = ResidualBlock(curr_channels, curr_channels, emb_dim)
        self.bottleneck2 = ResidualBlock(curr_channels, curr_channels, emb_dim)

        self.up_layers = nn.ModuleList()
        for mult in reversed(channel_mults):
            out_channels = base_channels * mult
            #self.up_layers.append(nn.ConvTranspose2d(curr_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1))
            self.up_layers.append(nn.Conv2d(curr_channels, out_channels, kernel_size = 3, stride = 1, padding = 1))
            self.up_layers.append(ResidualBlock(out_channels + out_channels, out_channels, emb_dim))
            self.up_layers.append(ResidualBlock(out_channels, out_channels, emb_dim))
            curr_channels = out_channels
        
        self.final_conv = nn.Conv2d(curr_channels, in_channels, 3, 1, 1)
    
    def forward(self, x, t):
        t_emb = self.emb(t)
        x = self.first_conv(x)
        x = self.activation(x)

        res_storage = []
        for i in range(0, len(self.down_layers), 3):
            x = self.down_layers[i](x, t_emb)
            x = self.down_layers[i+1](x, t_emb)
            res_storage.append(x)
            x = self.down_layers[i+2](x)
            x = self.activation(x)

        x = self.bottleneck1(x, t_emb)
        x = self.bottleneck2(x, t_emb)

        for j in range(0, len(self.up_layers), 3):
            res = res_storage.pop()
            #x = self.up_layers[j](x)
            x = F.interpolate(x, scale_factor = 2, mode = 'bilinear', align_corners=False)
            x = self.up_layers[j](x)
            x = self.activation(x)
            x = torch.cat([x, res], dim=1)
            x = self.up_layers[j+1](x, t_emb)
            x = self.up_layers[j+2](x, t_emb)

        return self.final_conv(x)