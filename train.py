import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as T
import matplotlib.pyplot as plt
from diffusion import Diffusion
from train_one_epoch import train_one_epoch

num_channels = 1
train_transforms = T.Compose([
    T.ToTensor(),
    T.Grayscale(num_output_channels = num_channels),
    T.CenterCrop(128),
    T.Normalize(mean = [0.5]*num_channels, std = [0.5]*num_channels) #go from [0, 1] to [-1, 1]
])

dataset = datasets.CelebA(root = 'data', split = 'train', download = False, transform = train_transforms)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

device = torch.device('cuda:0')

beta = torch.linspace(10**-4, 0.02, 1000)
beta = torch.cat([torch.tensor([0]), beta], dim = 0)
beta = beta.to(device)

model = Diffusion(beta = beta, noise_model = 'UNet', num_channels = num_channels, image_height = 128, image_width = 128)

SAVE_PATH = 'custom_mha_models/Attention_Diffusion_Model_'
LOAD_PATH = 'custom_mha_models/Attention_Diffusion_Model_9.pth'

checkpoint = torch.load(LOAD_PATH)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = 2e-5)
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 1)
gradscaler = torch.amp.GradScaler()

for epoch in range(10, 15):
    train_one_epoch(model, dataloader, optimizer, device, epoch, print_freq = 50, lr_update_freq = 200, lr_scheduler = lr_scheduler, gradscaler = gradscaler, grad_clip_norm_max = 1.0)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, SAVE_PATH + str(epoch) + '.pth')

    with torch.no_grad():
        model.eval()
        output = model.sample(8)
        #output = output.squeeze()
        output = (output + 1.0)/2.0
        
        torchvision.utils.save_image(output, f'attention_images/images{epoch}.png', nrow = 4)