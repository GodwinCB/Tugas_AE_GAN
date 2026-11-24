# Evaluasi test set (Q1), FID, latency GPU & CPU, dan tabel metrik komparatif
# Butuh semua checkpoint dari -> vanilla_ae, sparse_ae, vae, vanilla_gan, dcgan, ddpm

import copy
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from config import SEED, DEVICE, LATENT_DIM, RESULT_DIR
from data import get_loaders, get_test_tensors
from metrics import get_metrics, generate_all, compute_fid, measure_latency
from utils import set_seed, ensure_dirs, load_checkpoint, load_json, save_json
from vanilla_ae import VanillaAE
from sparse_ae import SparseAE
from vae import VAE, sample_vae
from gan_common import Z_DIM, sample_gan
from vanilla_gan import VanillaGenerator
from dcgan import DCGenerator
from ddpm import UNet, sample_ddpm

def main():
    ensure_dirs()
    set_seed()
    _, _, test_loader = get_loaders()
    x_test_all, _ = get_test_tensors(test_loader)

    ae = load_checkpoint(VanillaAE(), 'vanilla_ae')
    sparse_ae = load_checkpoint(SparseAE(), 'sparse_ae')
    vae = load_checkpoint(VAE(), 'vae')
    G_van = load_checkpoint(VanillaGenerator(), 'vanilla_gan_G')
    G_dc = load_checkpoint(DCGenerator(), 'dcgan_G')
    unet = load_checkpoint(UNet(), 'ddpm_unet')

    # Q1 -> Vanilla AE vs Sparse AE vs VAE (Test Set)
    with torch.no_grad():
        x_dev = x_test_all.to(DEVICE)

        recon_ae = ae(x_dev).cpu()
        recon_sparse = sparse_ae(x_dev)[0].cpu()

        mu_test, _ = vae.encode(x_dev)
        recon_vae = vae.decode(mu_test).cpu() # Rekonstruksi VAE pake mu

        h_vanilla = F.relu(ae.fc1(x_dev)).cpu()
        h_sparse = sparse_ae.encode(x_dev)[1].cpu()

    metrics_ae = get_metrics(x_test_all, recon_ae)
    metrics_sparse = get_metrics(x_test_all, recon_sparse)
    metrics_vae = get_metrics(x_test_all, recon_vae)

    print(f'{"Model":<20}{"MSE":>10}{"PSNR (dB)":>12}{"SSIM":>10}')
    print(f'{"Vanilla AE":<20}{metrics_ae[0]:>10.5f}{metrics_ae[1]:>12.2f}{metrics_ae[2]:>10.4f}')
    print(f'{"Sparse AE":<20}{metrics_sparse[0]:>10.5f}{metrics_sparse[1]:>12.2f}{metrics_sparse[2]:>10.4f}')
    print(f'{"VAE":<20}{metrics_vae[0]:>10.5f}{metrics_vae[1]:>12.2f}{metrics_vae[2]:>10.4f}')

    zero_vanilla = (h_vanilla <= 1e-6).float().mean().item()
    zero_sparse = (h_sparse <= 1e-6).float().mean().item()
    print(f'Hidden zero-fraction  Vanilla AE: {zero_vanilla:.2%}, Sparse AE: {zero_sparse:.2%}')
    print(f'Hidden mean activation Vanilla AE: {h_vanilla.mean():.4f}, Sparse AE: {h_sparse.mean():.4f}')

    # Sampling & FID (VAE, Vanilla GAN, DCGAN, DDPM)
    N_FID = len(x_test_all)

    torch.manual_seed(SEED)
    fake_vae = generate_all(lambda n: sample_vae(vae, n), N_FID)
    fake_van = generate_all(lambda n: sample_gan(G_van, n), N_FID)
    fake_dc = generate_all(lambda n: sample_gan(G_dc, n), N_FID)

    fid_vae = compute_fid(x_test_all, fake_vae)
    fid_van = compute_fid(x_test_all, fake_van)
    fid_dc = compute_fid(x_test_all, fake_dc)

    torch.manual_seed(SEED)
    fake_ddpm = generate_all(lambda n: sample_ddpm(unet, n)[0].view(n, -1), N_FID)
    fid_ddpm = compute_fid(x_test_all, fake_ddpm)

    x64 = x_test_all[:64].to(DEVICE)

    lat_ae = measure_latency(lambda: ae(x64))
    lat_sparse = measure_latency(lambda: sparse_ae(x64))
    lat_vae = measure_latency(lambda: sample_vae(vae, 64))
    lat_van = measure_latency(lambda: sample_gan(G_van, 64))
    lat_dc = measure_latency(lambda: sample_gan(G_dc, 64))

    # DDPM
    sample_ddpm(unet, 64)
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(3):
        sample_ddpm(unet, 64)
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    lat_ddpm = (time.time() - start) / 3 * 1000

    print(f'Latency DDPM / DCGAN: {lat_ddpm / lat_dc:.0f}x, DDPM / Vanilla GAN: {lat_ddpm / lat_van:.0f}x')

    # Latency CPU
    cpu_vae = copy.deepcopy(vae).cpu().eval()
    cpu_van = copy.deepcopy(G_van).cpu().eval()
    cpu_dc = copy.deepcopy(G_dc).cpu().eval()

    cpu_lat = {
        'VAE': measure_latency(lambda: cpu_vae.decode(torch.randn(64, LATENT_DIM))),
        'Vanilla GAN': measure_latency(lambda: cpu_van(torch.randn(64, Z_DIM))),
        'DCGAN': measure_latency(lambda: cpu_dc(torch.randn(64, Z_DIM))),
    }
    print('CPU latency (ms / batch 64) - ' + ', '.join(f'{k}: {v:.2f}' for k, v in cpu_lat.items()))

    # Tabel Metrik Komparatif
    t_ae = np.mean(load_json('vanilla_ae')['epoch_times'])
    t_sparse = np.mean(load_json('sparse_ae')['epoch_times'])
    t_vae = np.mean(load_json('vae')['epoch_times'])
    t_van = np.mean(load_json('vanilla_gan')['time'])
    t_dc = np.mean(load_json('dcgan')['time'])
    t_ddpm = np.mean(load_json('ddpm')['epoch_times'])

    NA = 'N/A'
    results = pd.DataFrame([
        ['Vanilla AE', 64, t_ae, lat_ae, *metrics_ae, NA],
        ['Sparse AE', '64 (256 sparse)', t_sparse, lat_sparse, *metrics_sparse, NA],
        ['VAE', 64, t_vae, lat_vae, *metrics_vae, fid_vae],
        ['Vanilla GAN', 100, t_van, lat_van, NA, NA, NA, fid_van],
        ['DCGAN', 100, t_dc, lat_dc, NA, NA, NA, fid_dc],
        ['DDPM', '28x28', t_ddpm, lat_ddpm, NA, NA, NA, fid_ddpm],
    ], columns = ['Model', 'Dim Laten', 'Time/Epoch (s)', 'Latency (ms/batch 64)', 'Test MSE', 'Test PSNR (dB)', 'Test SSIM', 'FID'])

    results.to_csv(f'{RESULT_DIR}/results.csv', index = False)
    print(results.to_string(index = False, float_format = lambda v: f'{v:.4g}'))

    save_json('evaluate', {
        'metrics': {'vanilla_ae': metrics_ae, 'sparse_ae': metrics_sparse, 'vae': metrics_vae},
        'hidden_zero_fraction': {'vanilla_ae': zero_vanilla, 'sparse_ae': zero_sparse},
        'fid': {'vae': fid_vae, 'vanilla_gan': fid_van, 'dcgan': fid_dc, 'ddpm': fid_ddpm},
        'latency_gpu_ms': {'vanilla_ae': lat_ae, 'sparse_ae': lat_sparse, 'vae': lat_vae,
                           'vanilla_gan': lat_van, 'dcgan': lat_dc, 'ddpm': lat_ddpm},
        'latency_cpu_ms': cpu_lat,
    })

if __name__ == '__main__':
    main()