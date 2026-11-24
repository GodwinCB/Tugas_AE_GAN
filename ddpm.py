# DDPM -> Noise Schedule, U-Net, Training, Sampling

import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import SEED, DEVICE
from data import get_loaders
from utils import set_seed, ensure_dirs, save_checkpoint, save_json

T_STEPS = 1000
TIME_DIM = 128
DDPM_EPOCHS = 40
DDPM_LR = 0.0005

betas = torch.linspace(1e-4, 0.02, T_STEPS, device = DEVICE)
alphas = 1 - betas
alpha_bar = torch.cumprod(alphas, dim = 0)

def time_embedding(t):
    half = TIME_DIM // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device = t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim = 1)

class Block(nn.Module):
    def __init__(self, c_in, c_out):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding = 1)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding = 1)
        self.norm1 = nn.GroupNorm(8, c_out)
        self.norm2 = nn.GroupNorm(8, c_out)
        self.time = nn.Linear(TIME_DIM, c_out)

    def forward(self, x, t):
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.time(t)[:, :, None, None]
        return F.silu(self.norm2(self.conv2(h)))

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.time_mlp = nn.Sequential(nn.Linear(TIME_DIM, TIME_DIM), nn.SiLU(), nn.Linear(TIME_DIM, TIME_DIM))
        self.b1 = Block(1, 64)
        self.b2 = Block(64, 128)
        self.b3 = Block(128, 256)
        self.b4 = Block(256 + 128, 128)
        self.b5 = Block(128 + 64, 64)
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x, t):
        t = self.time_mlp(time_embedding(t))
        h1 = self.b1(x, t)
        h2 = self.b2(F.max_pool2d(h1, 2), t)
        h3 = self.b3(F.max_pool2d(h2, 2), t)
        u = self.b4(torch.cat([F.interpolate(h3, scale_factor = 2), h2], dim = 1), t)
        u = self.b5(torch.cat([F.interpolate(u, scale_factor = 2), h1], dim = 1), t)
        return self.out(u)

@torch.no_grad()
def sample_ddpm(unet, n, save_at = ()):
    unet.eval()
    x = torch.randn(n, 1, 28, 28, device = DEVICE)
    snapshots = {}

    for t in reversed(range(T_STEPS)):
        eps = unet(x, torch.full((n,), t, device = DEVICE))
        x = (x - betas[t] / (1 - alpha_bar[t]).sqrt() * eps) / alphas[t].sqrt()
        if t > 0:
            x = x + betas[t].sqrt() * torch.randn_like(x)

        if t in save_at:
            snapshots[t] = ((x.clamp(-1, 1) + 1) / 2).cpu()

    return (x.clamp(-1, 1) + 1) / 2, snapshots

def main():
    ensure_dirs()
    set_seed()
    train_loader, _, _ = get_loaders()

    torch.manual_seed(SEED)
    unet = UNet().to(DEVICE)
    optimizer_ddpm = torch.optim.Adam(unet.parameters(), lr = DDPM_LR)

    ddpm_loss_history = []
    ddpm_epoch_times = []

    for epoch in range(DDPM_EPOCHS):
        unet.train()
        epoch_loss = []
        start = time.time()

        for x, _ in train_loader:
            x0 = x.to(DEVICE) * 2 - 1
            t = torch.randint(0, T_STEPS, (x0.size(0),), device = DEVICE)
            noise = torch.randn_like(x0)

            ab = alpha_bar[t][:, None, None, None]
            xt = ab.sqrt() * x0 + (1 - ab).sqrt() * noise

            loss = F.mse_loss(unet(xt, t), noise)

            optimizer_ddpm.zero_grad()
            loss.backward()
            optimizer_ddpm.step()

            epoch_loss.append(loss.item())

        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        ddpm_epoch_times.append(time.time() - start)
        ddpm_loss_history.append(np.mean(epoch_loss))

        print(f'Epoch[{epoch + 1} / {DDPM_EPOCHS}], Noise MSE: {ddpm_loss_history[-1]:.5f}')

    save_checkpoint(unet, 'ddpm_unet')
    save_json('ddpm', {'loss': ddpm_loss_history, 'epoch_times': ddpm_epoch_times})
    print(f'Param U-Net: {sum(p.numel() for p in unet.parameters()):,}')

if __name__ == '__main__':
    main()