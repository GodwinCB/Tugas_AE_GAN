import time
import torch
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure as ssim_fn

from config import DEVICE, IMAGE_SIZE

def get_metrics(x, x_hat):

    mse_img = ((x - x_hat) ** 2).mean(dim = 1).clamp(min = 1e-10)

    mse = mse_img.mean().item()
    psnr = (10 * torch.log10(1.0 / mse_img)).mean().item()
    ssim = ssim_fn(x_hat.view(-1, 1, 28, 28), x.view(-1, 1, 28, 28), data_range = 1.0).item()

    return mse, psnr, ssim

def val_mse(recon_fn, loader):
    total = 0.0
    count = 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.view(-1, IMAGE_SIZE).to(DEVICE)
            total += F.mse_loss(recon_fn(x), x, reduction = 'sum').item()
            count += x.numel()
    return total / count

def generate_all(sample_fn, n_total, batch = 500):
    outs = []
    with torch.no_grad():
        for i in range(0, n_total, batch):
            outs.append(sample_fn(min(batch, n_total - i)).clamp(0, 1).cpu())
    return torch.cat(outs)

def compute_fid(x_real, fake_01):

    from torchmetrics.image.fid import FrechetInceptionDistance

    fid = FrechetInceptionDistance(feature = 2048, normalize = True).to(DEVICE)
    for source, is_real in [(x_real, True), (fake_01, False)]:
        for i in range(0, len(source), 500):
            batch = source[i:i + 500].view(-1, 1, 28, 28).repeat(1, 3, 1, 1).to(DEVICE)
            fid.update(batch, real = is_real)
    return fid.compute().item()

def measure_latency(fn, runs = 100):
    with torch.no_grad():
        for _ in range(10):
            fn()
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(runs):
            fn()
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
    return (time.time() - start) / runs * 1000