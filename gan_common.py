# Bagian yang dipake bareng Vanilla GAN & DCGAN

import time
import numpy as np
import torch
import torch.nn as nn

from config import DEVICE

Z_DIM = 100
GAN_EPOCHS = 50
GAN_LR = 0.0002

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)

def sample_gan(G, n):
    G.eval()
    z = torch.randn(n, Z_DIM, device = DEVICE)
    return ((G(z) + 1) / 2).view(n, -1)

def train_gan(G, D, train_loader, epochs):

    criterion = nn.BCEWithLogitsLoss()
    optimizer_g = torch.optim.Adam(G.parameters(), lr = GAN_LR, betas = (0.5, 0.999))
    optimizer_d = torch.optim.Adam(D.parameters(), lr = GAN_LR, betas = (0.5, 0.999))

    history = {'d_loss': [], 'g_loss': [], 'd_real': [], 'd_fake': [], 'time': []}

    for epoch in range(epochs):
        G.train()
        D.train()
        d_losses, g_losses, d_reals, d_fakes = [], [], [], []
        start = time.time()

        for x, _ in train_loader:
            x = x.to(DEVICE) * 2 - 1
            b = x.size(0)
            ones = torch.ones(b, 1, device = DEVICE)
            zeros = torch.zeros(b, 1, device = DEVICE)

            # Update Discriminator
            z = torch.randn(b, Z_DIM, device = DEVICE)
            fake = G(z)
            out_real = D(x)
            out_fake = D(fake.detach())
            loss_d = criterion(out_real, ones) + criterion(out_fake, zeros)

            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()

            # Update Generator
            loss_g = criterion(D(fake), ones)

            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()

            d_losses.append(loss_d.item())
            g_losses.append(loss_g.item())
            d_reals.append(torch.sigmoid(out_real).mean().item())
            d_fakes.append(torch.sigmoid(out_fake).mean().item())

        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        history['time'].append(time.time() - start)
        history['d_loss'].append(np.mean(d_losses))
        history['g_loss'].append(np.mean(g_losses))
        history['d_real'].append(np.mean(d_reals))
        history['d_fake'].append(np.mean(d_fakes))

        print(f'Epoch[{epoch + 1} / {epochs}], D Loss: {history["d_loss"][-1]:.4f}, G Loss: {history["g_loss"][-1]:.4f}, D(x): {history["d_real"][-1]:.3f}, D(G(z)): {history["d_fake"][-1]:.3f}')

    return history