# Generative Models Comparison (D6421 Sesi 07)

Vanilla AE, Sparse AE, VAE, Vanilla GAN, DCGAN, DDPM dengan Fashion-MNIST, plus MLP-classifier limitasi experimen (Tahap 1-5).

Setup: PyTorch, seed 42, split 80/10/10 over all 70k images, batch size 128.

## Run

pip install -r requirements.txt

# Tahap 1-5
python -m mlp_classifier
python -m vanilla_ae
python -m mlp_limits

# Model Lainnya
python -m sparse_ae
python -m vae
python -m vanilla_gan
python -m dcgan
python -m ddpm

# Komparasi
python -m evaluate
python -m figures