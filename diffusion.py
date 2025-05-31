import torch
import torch.nn as nn
from unet import UNet
from unet2 import UNet2

class Diffusion(nn.Module):
    def __init__(self, beta, noise_model, num_channels, image_height, image_width):
        super().__init__()
        self.T = len(beta) - 1
        self.beta = beta
        self.alpha = 1.0 - beta
        self.alphabar = torch.cumprod(self.alpha, dim = 0)

        if noise_model == 'UNet':
            self.npn = UNet(num_channels)
        else:
            self.npn = UNet2(num_channels)
            
        self.H = image_height
        self.W = image_width
        self.C = num_channels
        self.loss_fn = nn.MSELoss()

    def noise_pred(self, x, t):
        return self.npn(x, t)
    
    def sample(self, batch_size = 1): #call this if you want inference, batch_size = number of images you will get
        device = next(self.parameters()).device
        xt = torch.randn((batch_size, self.C, self.H, self.W), device = device)

        for t in range(self.T, 0, -1):
            if t > 1:
                z = torch.randn((batch_size, self.C, self.H, self.W), device = device)
            else:
                z = torch.zeros((batch_size, self.C, self.H, self.W), device = device)

            betatilde_t = (1 - self.alphabar[t-1])/(1 - self.alphabar[t]) * self.beta[t]
            eps_theta = self.noise_pred(xt, torch.full((batch_size, ), t, device = device))
            xt = (xt - (1-self.alpha[t])/(torch.sqrt(1-self.alphabar[t])) * eps_theta)/torch.sqrt(self.alpha[t]) + torch.sqrt(betatilde_t)*z
        
        return xt

    def forward(self, x0): #only for training. Takes clean images, adds noise, gets noise preds, outputs the loss directly since outside modules don't know the true noise
        batch = x0.shape[0]
        device = next(self.parameters()).device

        t = torch.randint(1, self.T+1, size = (batch, ), device = device)
        eps = torch.randn(x0.shape, device = device)

        alphabar_t = self.alphabar[t].view(batch, 1, 1, 1)
        xt = torch.sqrt(alphabar_t) * x0 + torch.sqrt(1 - alphabar_t) * eps

        eps_theta = self.noise_pred(xt, t)

        loss = self.loss_fn(eps, eps_theta)

        return loss