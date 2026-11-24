# Vanilla Autoencoder (784 -> 256 -> 64 -> 256 -> 784)

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

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(LATENT_DIM, HIDDEN_DIM_1)
        self.fc2 = nn.Linear(HIDDEN_DIM_1, IMAGE_SIZE)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        return torch.sigmoid(self.fc2(h))

class VanillaAE(nn.Module):
    def __init__(self):
        super(VanillaAE, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(IMAGE_SIZE, HIDDEN_DIM_1)
        self.fc2 = nn.Linear(HIDDEN_DIM_1, LATENT_DIM)

        # Decoder
        self.decoder = Decoder()

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h) # Latent linear (Tanpa ReLU)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))

def main():
    ensure_dirs()
    set_seed()
    train_loader, val_loader, _ = get_loaders()

    torch.manual_seed(SEED)
    ae = VanillaAE().to(DEVICE)
    optimizer_ae = torch.optim.Adam(ae.parameters(), lr = LEARNING_RATE)

    ae_loss_history = []
    ae_val_history = []
    ae_epoch_times = []
    best_val = float('inf')
    best_state_ae = None

    for epoch in range(MAX_EPOCHS):
        ae.train()
        epoch_loss = []
        start = time.time()

        for x, _ in train_loader:
            x = x.view(-1, IMAGE_SIZE).to(DEVICE)

            x_recon = ae(x)
            loss = F.mse_loss(x_recon, x)

            optimizer_ae.zero_grad()
            loss.backward()
            optimizer_ae.step()

            epoch_loss.append(loss.item())

        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        ae_epoch_times.append(time.time() - start)

        ae.eval()
        val_loss = val_mse(ae, val_loader)
        ae_loss_history.append(np.mean(epoch_loss))
        ae_val_history.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state_ae = copy.deepcopy(ae.state_dict())

        print(f'Epoch[{epoch + 1} / {MAX_EPOCHS}], Train MSE: {np.mean(epoch_loss):.5f}, Val MSE: {val_loss:.5f}')

    ae.load_state_dict(best_state_ae)
    save_checkpoint(ae, 'vanilla_ae')
    save_json('vanilla_ae', {'loss': ae_loss_history, 'val': ae_val_history, 'epoch_times': ae_epoch_times})
    print(f'Rata-rata waktu per epoch: {np.mean(ae_epoch_times):.2f} s')

if __name__ == '__main__':
    main()