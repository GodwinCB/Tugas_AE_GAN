# Tahap 2-5 + Figur 3 -> Bukti batasan MLP Classifier untuk image generation
# Butuh checkpoint dari mlp_classifier.py dan vanilla_ae.py

import copy
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from config import SEED, DEVICE, IMAGE_SIZE, LATENT_DIM, NUM_CLASSES, MAX_EPOCHS, LEARNING_RATE, CLASS_NAMES
from data import get_loaders, get_test_tensors
from metrics import get_metrics, val_mse
from mlp_classifier import MLPClassifier
from vanilla_ae import VanillaAE, Decoder
from utils import set_seed, ensure_dirs, load_checkpoint, save_checkpoint, save_json, save_fig

def main():
    ensure_dirs()
    set_seed()
    train_loader, val_loader, test_loader = get_loaders()
    x_test_all, y_test_all = get_test_tensors(test_loader)

    clf = load_checkpoint(MLPClassifier(), 'mlp_classifier')
    ae = load_checkpoint(VanillaAE(), 'vanilla_ae')
    results = {}

    W1 = clf.fc1.weight.detach().cpu()
    W2 = clf.fc2.weight.detach().cpu()
    W3 = clf.fc3.weight.detach().cpu()

    y_onehot = torch.eye(NUM_CLASSES)

    z_hat = F.relu(y_onehot @ W3)
    h1_hat = F.relu(z_hat @ W2)
    x_hat_class = torch.sigmoid(h1_hat @ W1)

    plt.figure(figsize=(20, 3))
    for index in range(NUM_CLASSES):
        ax = plt.subplot(1, NUM_CLASSES, index + 1)
        plt.imshow(x_hat_class[index].view(28, 28).numpy(), cmap = 'gray')
        plt.title(CLASS_NAMES[index])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.tight_layout()
    save_fig('tahap2_class_projection')

    x_test = x_test_all[:10]
    y_test = y_test_all[:10]

    with torch.no_grad():
        z_mlp = clf.features(x_test.to(DEVICE)).cpu()

    x_hat_latent = torch.sigmoid(F.relu(z_mlp @ W2) @ W1)
    mse_transpose = F.mse_loss(x_hat_latent, x_test).item()
    print(f'MSE Latent Transpose Projection: {mse_transpose:.4f}')

    number = 10
    plt.figure(figsize=(20, 5))
    for index in range(number):
        ax = plt.subplot(2, number, index + 1)
        plt.imshow(x_test[index].view(28, 28).numpy(), cmap = 'gray')
        plt.title(CLASS_NAMES[y_test[index]])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    for index in range(number):
        ax = plt.subplot(2, number, number + index + 1)
        plt.imshow(x_hat_latent[index].view(28, 28).numpy(), cmap = 'gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.tight_layout()
    save_fig('tahap3_latent_projection')

    # Baseline MSE
    mse_black = x_test_all.pow(2).mean().item()
    mse_gray = ((x_test_all - x_test_all.mean()) ** 2).mean().item()
    mse_mean_img = ((x_test_all - x_test_all.mean(dim = 0)) ** 2).mean().item()
    print(f'Baseline MSE - Black: {mse_black:.4f}, Flat Gray: {mse_gray:.4f}, Mean Image: {mse_mean_img:.4f}')

    with torch.no_grad():
        z_all = clf.features(x_test_all.to(DEVICE)).cpu()

    pre_sigmoid = F.relu(z_all @ W2) @ W1
    x_hat_all = torch.sigmoid(pre_sigmoid)
    mse_t3 = F.mse_loss(x_hat_all, x_test_all).item()

    print(f'MSE Latent Transpose Projection (full test): {mse_t3:.4f}')
    print(f'Pre-sigmoid min/max/mean/std: {pre_sigmoid.min():.1f} / {pre_sigmoid.max():.1f} / {pre_sigmoid.mean():.1f} / {pre_sigmoid.std():.1f}')
    print(f'Piksel saturasi (<0.01 atau >0.99): {((x_hat_all < 0.01) | (x_hat_all > 0.99)).float().mean():.2%}')

    results['baseline'] = {'black': mse_black, 'gray': mse_gray, 'mean_img': mse_mean_img}
    results['tahap3_mse_full_test'] = mse_t3

    for p in clf.parameters():
        p.requires_grad = False
    clf.eval()

    torch.manual_seed(SEED)
    dec = Decoder().to(DEVICE)
    optimizer_dec = torch.optim.Adam(dec.parameters(), lr = LEARNING_RATE)

    def frozen_recon(x):
        with torch.no_grad():
            z = clf.features(x)
        return dec(z)

    dec_loss_history = []
    dec_val_history = []
    best_val = float('inf')
    best_state_dec = None

    for epoch in range(MAX_EPOCHS):
        dec.train()
        epoch_loss = []

        for x, _ in train_loader:
            x = x.view(-1, IMAGE_SIZE).to(DEVICE)

            x_recon = frozen_recon(x)
            loss = F.mse_loss(x_recon, x)

            optimizer_dec.zero_grad()
            loss.backward()
            optimizer_dec.step()

            epoch_loss.append(loss.item())

        dec.eval()
        val_loss = val_mse(frozen_recon, val_loader)
        dec_loss_history.append(np.mean(epoch_loss))
        dec_val_history.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state_dec = copy.deepcopy(dec.state_dict())

        print(f'Epoch[{epoch + 1} / {MAX_EPOCHS}], Train MSE: {np.mean(epoch_loss):.5f}, Val MSE: {val_loss:.5f}')

    dec.load_state_dict(best_state_dec)
    save_checkpoint(dec, 'frozen_decoder')
    save_json('frozen_decoder', {'loss': dec_loss_history, 'val': dec_val_history})

    # Evaluasi test set -> Frozen Decoder (z_MLP) vs Vanilla AE (z_AE)
    dec.eval()
    with torch.no_grad():
        recon_ae = ae(x_test_all.to(DEVICE)).cpu()
        recon_frozen = frozen_recon(x_test_all.to(DEVICE)).cpu()

    metrics_ae = get_metrics(x_test_all, recon_ae)
    metrics_frozen = get_metrics(x_test_all, recon_frozen)

    print(f'{"Model":<28}{"MSE":>10}{"PSNR (dB)":>12}{"SSIM":>10}')
    print(f'{"Vanilla AE (z_AE)":<28}{metrics_ae[0]:>10.5f}{metrics_ae[1]:>12.2f}{metrics_ae[2]:>10.4f}')
    print(f'{"Frozen Decoder (z_MLP)":<28}{metrics_frozen[0]:>10.5f}{metrics_frozen[1]:>12.2f}{metrics_frozen[2]:>10.4f}')

    results['vanilla_ae'] = metrics_ae
    results['frozen_decoder'] = metrics_frozen

    plt.figure(figsize=(20, 6))
    rows = [x_test_all, recon_ae, recon_frozen]
    titles = ['Original', 'Vanilla AE', 'Frozen Decoder']
    for r in range(3):
        for index in range(number):
            ax = plt.subplot(3, number, r * number + index + 1)
            plt.imshow(rows[r][index].view(28, 28).numpy(), cmap = 'gray')
            if index == 0:
                ax.set_ylabel(titles[r])
            ax.set_xticks([])
            ax.set_yticks([])
    plt.tight_layout()
    save_fig('tahap4_frozen_vs_ae')

    # Mean & std per dimensi dari z_MLP test set
    mu_z = z_all.mean(dim = 0)
    sigma_z = z_all.std(dim = 0)

    torch.manual_seed(SEED)
    z_rand = mu_z + sigma_z * torch.randn(16, LATENT_DIM)

    with torch.no_grad():
        x_rand = dec(z_rand.to(DEVICE)).cpu()

    dead_dims = (z_all.max(dim = 0).values == 0).sum().item()
    neg_frac = (z_rand < 0).float().mean().item()
    print(f'Dimensi z_MLP yang selalu 0 di test set: {dead_dims} / {LATENT_DIM}')
    print(f'Proporsi elemen negatif pada z_rand: {neg_frac:.2%}')

    plt.figure(figsize=(6, 6))
    for index in range(16):
        ax = plt.subplot(4, 4, index + 1)
        plt.imshow(x_rand[index].view(28, 28).numpy(), cmap = 'gray')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.suptitle('Prior Sampling pada Ruang Laten MLP Classifier')
    plt.tight_layout()
    save_fig('tahap5_prior_sampling')

    # Cek mode coverage sampel Tahap 5
    torch.manual_seed(SEED)
    z_many = mu_z + sigma_z * torch.randn(1000, LATENT_DIM)

    with torch.no_grad():
        x_many = dec(z_many.to(DEVICE))
        pred = clf(x_many).argmax(dim = 1).cpu()

    coverage = dict(zip(CLASS_NAMES, torch.bincount(pred, minlength = NUM_CLASSES).tolist()))
    print(coverage)

    results['tahap5'] = {'dead_dims': dead_dims, 'neg_frac': neg_frac, 'mode_coverage': coverage}
    save_json('mlp_limits', results)

    # Figure 3
    rows = [
        ('(1) Ground Truth', x_test_all[:10]),
        ('(2a) W^T dari y_c', x_hat_class),
        ('(2b) W^T dari z_MLP', x_hat_all[:10]),
        ('(3a) Frozen Decoder', recon_frozen[:10]),
        ('(3b) Vanilla AE', recon_ae[:10]),
        ('(4) Prior Sampling', x_rand[:10]),
    ]

    plt.figure(figsize = (20, 12))
    for r, (name, imgs) in enumerate(rows):
        for index in range(10):
            ax = plt.subplot(len(rows), 10, r * 10 + index + 1)
            plt.imshow(imgs[index].view(28, 28).numpy(), cmap = 'gray')
            ax.set_xticks([])
            ax.set_yticks([])
            if index == 0:
                ax.set_ylabel(name)
            if r == 0:
                ax.set_title(CLASS_NAMES[y_test_all[index]]) 
            if r == 1:
                ax.set_title(CLASS_NAMES[index])
    plt.tight_layout()
    save_fig('figur3_mlp_limits')

if __name__ == '__main__':
    main()