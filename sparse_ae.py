# Sparse Autoencoder (784 -> 256 [L1] -> 64 -> 256 -> 784)

import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import SEED, DEVICE, IMAGE_SIZE, HIDDEN_DIM_1, LATENT_DIM, MAX_EPOCHS, LEARNING_RATE
from data import get_loaders
from metrics import val_mse
from utils import set_seed, ensure_dirs, save_checkpoint, save_json
from vanilla_ae import Decoder

LAMBDA_SPARSE = 1e-3

class SparseAE(nn.Module):
    def __init__(self):
        super(SparseAE, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(IMAGE_SIZE, HIDDEN_DIM_1)
        self.fc2 = nn.Linear(HIDDEN_DIM_1, LATENT_DIM)

        # Decoder
        self.decoder = Decoder()

    def encode(self, x):
        h = F.relu(self.fc1(x))
        z = self.fc2(h)
        return z, h

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z, h = self.encode(x)
        return self.decode(z), h

def main():
    ensure_dirs()
    set_seed()
    train_loader, val_loader, _ = get_loaders()

    torch.manual_seed(SEED)
    sparse_ae = SparseAE().to(DEVICE)
    optimizer_sparse = torch.optim.Adam(sparse_ae.parameters(), lr = LEARNING_RATE)

    sparse_loss_history = [] # Total loss
    sparse_recon_history = [] # MSE aja
    sparse_l1_history = [] # Penalty L1
    sparse_val_history = []
    sparse_epoch_times = []
    best_val = float('inf')
    best_state_sparse = None

    for epoch in range(MAX_EPOCHS):
        sparse_ae.train()
        epoch_loss = []
        epoch_recon = []
        epoch_l1 = []
        start = time.time()

        for x, _ in train_loader:
            x = x.view(-1, IMAGE_SIZE).to(DEVICE)

            x_recon, h = sparse_ae(x)
            recon_loss = F.mse_loss(x_recon, x)
            l1_loss = h.abs().mean()
            loss = recon_loss + LAMBDA_SPARSE * l1_loss

            optimizer_sparse.zero_grad()
            loss.backward()
            optimizer_sparse.step()

            epoch_loss.append(loss.item())
            epoch_recon.append(recon_loss.item())
            epoch_l1.append(l1_loss.item())

        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        sparse_epoch_times.append(time.time() - start)

        sparse_ae.eval()
        val_loss = val_mse(lambda x: sparse_ae(x)[0], val_loader) # Val pake MSE
        sparse_loss_history.append(np.mean(epoch_loss))
        sparse_recon_history.append(np.mean(epoch_recon))
        sparse_l1_history.append(np.mean(epoch_l1))
        sparse_val_history.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state_sparse = copy.deepcopy(sparse_ae.state_dict())

        print(f'Epoch[{epoch + 1} / {MAX_EPOCHS}], Recon MSE: {np.mean(epoch_recon):.5f}, L1: {np.mean(epoch_l1):.4f}, Val MSE: {val_loss:.5f}')

    sparse_ae.load_state_dict(best_state_sparse)
    save_checkpoint(sparse_ae, 'sparse_ae')
    save_json('sparse_ae', {'loss': sparse_loss_history, 'recon': sparse_recon_history, 'l1': sparse_l1_history,
                            'val': sparse_val_history, 'epoch_times': sparse_epoch_times})
    print(f'Rata-rata waktu per epoch: {np.mean(sparse_epoch_times):.2f} s')

if __name__ == '__main__':
    main()