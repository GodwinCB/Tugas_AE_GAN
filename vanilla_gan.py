# Vanilla GAN (fully connected)

import torch
import torch.nn as nn

from config import SEED, DEVICE, IMAGE_SIZE
from data import get_loaders
from gan_common import Z_DIM, GAN_EPOCHS, train_gan
from utils import set_seed, ensure_dirs, save_checkpoint, save_json

class VanillaGenerator(nn.Module):
    def __init__(self):
        super(VanillaGenerator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(Z_DIM, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, IMAGE_SIZE), nn.Tanh()
        )

    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)

class VanillaDiscriminator(nn.Module):
    def __init__(self):
        super(VanillaDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(IMAGE_SIZE, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x)

def main():
    ensure_dirs()
    set_seed()
    train_loader, _, _ = get_loaders()

    torch.manual_seed(SEED)
    G_van = VanillaGenerator().to(DEVICE)
    D_van = VanillaDiscriminator().to(DEVICE)
    hist_van = train_gan(G_van, D_van, train_loader, GAN_EPOCHS)

    save_checkpoint(G_van, 'vanilla_gan_G')
    save_json('vanilla_gan', hist_van)
    print(f'Param Generator - Vanilla GAN: {sum(p.numel() for p in G_van.parameters()):,}')

if __name__ == '__main__':
    main()