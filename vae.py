# Variational Autoencoder (784 -> 256 -> mu/log_var 64 -> 256 -> 784)

import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import SEED, DEVICE, IMAGE_SIZE, HIDDEN_DIM_1, LATENT_DIM, MAX_EPOCHS, LEARNING_RATE
from data import get_loaders
from utils import set_seed, ensure_dirs, save_checkpoint, save_json
from vanilla_ae import Decoder

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(IMAGE_SIZE, HIDDEN_DIM_1)
        self.fc2 = nn.Linear(HIDDEN_DIM_1, LATENT_DIM) # Ngitung mean (miu)
        self.fc3 = nn.Linear(HIDDEN_DIM_1, LATENT_DIM) # Log variance

        # Decoder
        self.decoder = Decoder()

    def encode(self, x):
        h = F.relu(self.fc1(x))
        log_var = torch.clamp(self.fc3(h), -10.0, 10.0)
        return self.fc2(h), log_var

    # Reparameterization
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var) # Ngitung std
        eps = torch.randn_like(std) # Ngitung epsilon -> random
        return mu + eps * std # Nilai z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

def vae_loss(x, x_recon, mu, log_var):
    recon = F.binary_cross_entropy(x_recon, x, reduction = 'sum') / x.size(0)
    kl = (-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim = 1)).mean()
    return recon, kl

def sample_vae(vae, n):
    z = torch.randn(n, LATENT_DIM, device = DEVICE)
    return vae.decode(z).view(n, -1)

def main():
    ensure_dirs()
    set_seed()
    train_loader, val_loader, _ = get_loaders()

    torch.manual_seed(SEED)
    vae = VAE().to(DEVICE)
    optimizer_vae = torch.optim.Adam(vae.parameters(), lr = LEARNING_RATE)

    vae_loss_history = []
    vae_recon_history = []
    vae_kl_history = []
    vae_val_history = []
    vae_epoch_times = []
    best_val = float('inf')
    best_state_vae = None

    for epoch in range(MAX_EPOCHS):
        vae.train()
        epoch_loss = []
        epoch_recon = []
        epoch_kl = []
        start = time.time()

        for x, _ in train_loader:
            x = x.view(-1, IMAGE_SIZE).to(DEVICE)

            x_recon, mu, log_var = vae(x)
            recon_loss, kl_div = vae_loss(x, x_recon, mu, log_var)
            loss = recon_loss + kl_div

            optimizer_vae.zero_grad()
            loss.backward()
            optimizer_vae.step()

            epoch_loss.append(loss.item())
            epoch_recon.append(recon_loss.item())
            epoch_kl.append(kl_div.item())

        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        vae_epoch_times.append(time.time() - start)

        vae.eval()
        val_total = 0.0
        val_count = 0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.view(-1, IMAGE_SIZE).to(DEVICE)
                x_recon, mu, log_var = vae(x)
                r, k = vae_loss(x, x_recon, mu, log_var)
                val_total += (r + k).item() * x.size(0)
                val_count += x.size(0)
        val_loss = val_total / val_count

        vae_loss_history.append(np.mean(epoch_loss))
        vae_recon_history.append(np.mean(epoch_recon))
        vae_kl_history.append(np.mean(epoch_kl))
        vae_val_history.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state_vae = copy.deepcopy(vae.state_dict())

        print(f'Epoch[{epoch + 1} / {MAX_EPOCHS}], Recon Loss: {np.mean(epoch_recon):.2f}, KL Div: {np.mean(epoch_kl):.2f}, Val Loss: {val_loss:.2f}')

    vae.load_state_dict(best_state_vae)
    save_checkpoint(vae, 'vae')
    save_json('vae', {'loss': vae_loss_history, 'recon': vae_recon_history, 'kl': vae_kl_history,
                      'val': vae_val_history, 'epoch_times': vae_epoch_times})
    print(f'Rata-rata waktu per epoch: {np.mean(vae_epoch_times):.2f} s')

if __name__ == '__main__':
    main()