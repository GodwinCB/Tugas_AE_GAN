# DCGAN (fully convolutional, strided conv, BatchNorm, tanh)

import torch
import torch.nn as nn

from config import SEED, DEVICE
from data import get_loaders
from gan_common import Z_DIM, GAN_EPOCHS, init_weights, train_gan
from utils import set_seed, ensure_dirs, save_checkpoint, save_json

class DCGenerator(nn.Module):
    def __init__(self):
        super(DCGenerator, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(Z_DIM, 256, 7, 1, 0, bias = False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z.view(-1, Z_DIM, 1, 1))

class DCDiscriminator(nn.Module):
    def __init__(self):
        super(DCDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1, bias = False),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, 7, 1, 0)
        )

    def forward(self, x):
        return self.net(x).view(-1, 1)

def main():
    ensure_dirs()
    set_seed()
    train_loader, _, _ = get_loaders()

    torch.manual_seed(SEED)
    G_dc = DCGenerator().to(DEVICE)
    D_dc = DCDiscriminator().to(DEVICE)
    G_dc.apply(init_weights)
    D_dc.apply(init_weights)
    hist_dc = train_gan(G_dc, D_dc, train_loader, GAN_EPOCHS)

    save_checkpoint(G_dc, 'dcgan_G')
    save_json('dcgan', hist_dc)
    print(f'Param Generator - DCGAN: {sum(p.numel() for p in G_dc.parameters()):,}')

if __name__ == '__main__':
    main()