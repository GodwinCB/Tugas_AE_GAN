import torch

SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 128
IMAGE_SIZE = 28 * 28
HIDDEN_DIM_1 = 256
LATENT_DIM = 64
NUM_CLASSES = 10
MAX_EPOCHS = 30
LEARNING_RATE = 0.001

CLASS_NAMES = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']

CHECKPOINT_DIR = './checkpoints'
RESULT_DIR = './results'
FIGURE_DIR = './figures'