# Figure 1, 2, 4, 5 + plot diagnostik
# Butuh semua checkpoint + history json

import torch
import matplotlib.pyplot as plt

from config import SEED, DEVICE
from data import get_loaders, get_test_tensors
from metrics import generate_all
from utils import set_seed, ensure_dirs, load_checkpoint, load_json, save_fig
from vanilla_ae import VanillaAE
from sparse_ae import SparseAE
from vae import VAE, sample_vae
from gan_common import Z_DIM, sample_gan
from vanilla_gan import VanillaGenerator
from dcgan import DCGenerator
from ddpm import UNet, sample_ddpm

def plot_mlp_history():
    h = load_json('mlp_classifier')
    plt.figure(figsize=(20, 4))

    plt.subplot(1, 2, 1)
    plt.plot(h['loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('MLP Classifier Train Loss History')

    plt.subplot(1, 2, 2)
    plt.plot(h['val_acc'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('MLP Classifier Validation Accuracy History')

    save_fig('mlp_classifier_history')

def plot_gan_diagnostics(hist_van, hist_dc):
    plt.figure(figsize = (20, 8))

    for row, (name, hist) in enumerate([('Vanilla GAN', hist_van), ('DCGAN', hist_dc)]):
        plt.subplot(2, 2, row * 2 + 1)
        plt.plot(hist['d_loss'], label = 'D Loss')
        plt.plot(hist['g_loss'], label = 'G Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{name} Loss History')
        plt.legend()

        plt.subplot(2, 2, row * 2 + 2)
        plt.plot(hist['d_real'], label = 'D(x)')
        plt.plot(hist['d_fake'], label = 'D(G(z))')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Probability')
        plt.title(f'{name} Discriminator Output')
        plt.legend()

    save_fig('gan_diagnostics')

def figur1_reconstruction(x_test_all, ae, sparse_ae, vae):
    with torch.no_grad():
        x_dev = x_test_all[:10].to(DEVICE)
        recon_ae = ae(x_dev).cpu()
        recon_sparse = sparse_ae(x_dev)[0].cpu()
        recon_vae = vae.decode(vae.encode(x_dev)[0]).cpu() # VAE pake mu (deterministik)

    number = 10
    rows = [x_test_all[:10], recon_ae, recon_sparse, recon_vae]
    titles = ['Original', 'Vanilla AE', 'Sparse AE', 'VAE']

    plt.figure(figsize=(20, 8))
    for r in range(4):
        for index in range(number):
            ax = plt.subplot(4, number, r * number + index + 1)
            plt.imshow(rows[r][index].view(28, 28).numpy(), cmap = 'gray')
            if index == 0:
                ax.set_ylabel(titles[r])
            ax.set_xticks([])
            ax.set_yticks([])
    plt.tight_layout()
    save_fig('figur1_reconstruction')

def plot_hidden_filters(ae, sparse_ae):
    fig, axes = plt.subplots(2, 16, figsize = (20, 3))
    for row, (name, fc) in enumerate([('Vanilla AE', ae.fc1), ('Sparse AE', sparse_ae.fc1)]):
        for i in range(16):
            axes[row, i].imshow(fc.weight[i].detach().cpu().view(28, 28), cmap = 'gray')
            axes[row, i].axis('off')
        axes[row, 0].set_title(name, loc = 'left')
    save_fig('hidden_filters')

def figur2_sampling(vae, G_van, G_dc, unet):
    # Panel A-C
    torch.manual_seed(SEED)
    panels = [('A: VAE', generate_all(lambda n: sample_vae(vae, n), 16)),
              ('B: Vanilla GAN', generate_all(lambda n: sample_gan(G_van, n), 16)),
              ('C: DCGAN', generate_all(lambda n: sample_gan(G_dc, n), 16))]

    plt.figure(figsize = (15, 5))
    for p, (title, samples) in enumerate(panels):
        for index in range(16):
            ax = plt.subplot(4, 12, (index // 4) * 12 + p * 4 + (index % 4) + 1)
            plt.imshow(samples[index].view(28, 28).numpy(), cmap = 'gray')
            ax.set_xticks([])
            ax.set_yticks([])
            if index == 1:
                ax.set_title(title, loc = 'left')
    plt.tight_layout()
    save_fig('figur2_panel_abc')

    # Panel D + Figure 5
    SAVE_AT = (999, 800, 600, 400, 200, 100, 50, 0)
    torch.manual_seed(SEED)
    samples, snaps = sample_ddpm(unet, 16, save_at = SAVE_AT)

    plt.figure(figsize = (5, 5))
    for index in range(16):
        ax = plt.subplot(4, 4, index + 1)
        plt.imshow(samples[index].view(28, 28).cpu().numpy(), cmap = 'gray')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.suptitle('D: DDPM')
    plt.tight_layout()
    save_fig('figur2_panel_d')

    plt.figure(figsize = (16, 8))
    for row in range(4):
        for col, t in enumerate(SAVE_AT):
            ax = plt.subplot(4, len(SAVE_AT), row * len(SAVE_AT) + col + 1)
            plt.imshow(snaps[t][row].view(28, 28).numpy(), cmap = 'gray')
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f't = {t}')
    plt.tight_layout()
    save_fig('figur5_ddpm_trajectory')

def figur4_interpolation(x_test_all, y_test_all, ae, sparse_ae, vae, G_van, G_dc):
    CLASS_A = 1
    CLASS_B = 7 
    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    idx_a = (y_test_all == CLASS_A).nonzero()[0].item()
    idx_b = (y_test_all == CLASS_B).nonzero()[0].item()
    x_a = x_test_all[idx_a:idx_a + 1].to(DEVICE)
    x_b = x_test_all[idx_b:idx_b + 1].to(DEVICE)

    torch.manual_seed(SEED)
    zg_a = torch.randn(1, Z_DIM, device = DEVICE)
    zg_b = torch.randn(1, Z_DIM, device = DEVICE)

    with torch.no_grad():
        models = [
            ('Vanilla AE', ae.encode(x_a), ae.encode(x_b), ae.decode),
            ('Sparse AE', sparse_ae.encode(x_a)[0], sparse_ae.encode(x_b)[0], sparse_ae.decode),
            ('VAE (mu)', vae.encode(x_a)[0], vae.encode(x_b)[0], vae.decode),
            ('Vanilla GAN (z acak)', zg_a, zg_b, lambda z: (G_van(z) + 1) / 2),
            ('DCGAN (z acak)', zg_a, zg_b, lambda z: (G_dc(z) + 1) / 2),
        ]

        plt.figure(figsize = (12, 10))
        for r, (name, za, zb, decode) in enumerate(models):
            for c, a in enumerate(alphas):
                ax = plt.subplot(len(models), len(alphas), r * len(alphas) + c + 1)
                plt.imshow(decode((1 - a) * za + a * zb).view(28, 28).cpu().numpy(), cmap = 'gray')
                ax.set_xticks([])
                ax.set_yticks([])
                if c == 0:
                    ax.set_ylabel(name)
                if r == 0:
                    ax.set_title(f'alpha = {a}')
        plt.tight_layout()
        save_fig('figur4_interpolation')

def figur5_loss_curves():
    h_ae, h_sparse, h_vae = load_json('vanilla_ae'), load_json('sparse_ae'), load_json('vae')
    h_van, h_dc, h_ddpm = load_json('vanilla_gan'), load_json('dcgan'), load_json('ddpm')

    curves = [
        ('Vanilla AE', {'Train MSE': h_ae['loss'], 'Val MSE': h_ae['val']}),
        ('Sparse AE', {'Train MSE': h_sparse['recon'], 'Val MSE': h_sparse['val']}),
        ('VAE', {'Train Loss (BCE+KL)': h_vae['loss'], 'Val Loss': h_vae['val']}),
        ('Vanilla GAN', {'D Loss': h_van['d_loss'], 'G Loss': h_van['g_loss']}),
        ('DCGAN', {'D Loss': h_dc['d_loss'], 'G Loss': h_dc['g_loss']}),
        ('DDPM', {'Noise MSE': h_ddpm['loss']}),
    ]

    fig, axes = plt.subplots(2, 3, figsize = (20, 8))
    for ax, (name, lines) in zip(axes.flat, curves):
        for label, values in lines.items():
            ax.plot(values, label = label)
        ax.set_title(name)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
    plt.tight_layout()
    save_fig('figur5_loss_curves')

def main():
    ensure_dirs()
    set_seed()
    _, _, test_loader = get_loaders()
    x_test_all, y_test_all = get_test_tensors(test_loader)

    ae = load_checkpoint(VanillaAE(), 'vanilla_ae')
    sparse_ae = load_checkpoint(SparseAE(), 'sparse_ae')
    vae = load_checkpoint(VAE(), 'vae')
    G_van = load_checkpoint(VanillaGenerator(), 'vanilla_gan_G')
    G_dc = load_checkpoint(DCGenerator(), 'dcgan_G')
    unet = load_checkpoint(UNet(), 'ddpm_unet')

    plot_mlp_history()
    plot_gan_diagnostics(load_json('vanilla_gan'), load_json('dcgan'))
    figur1_reconstruction(x_test_all, ae, sparse_ae, vae)
    plot_hidden_filters(ae, sparse_ae)
    figur2_sampling(vae, G_van, G_dc, unet)
    figur4_interpolation(x_test_all, y_test_all, ae, sparse_ae, vae, G_van, G_dc)
    figur5_loss_curves()

if __name__ == '__main__':
    main()