import torch
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torchvision import datasets, transforms

from config import SEED, BATCH_SIZE, IMAGE_SIZE

def get_loaders():
    transform = transforms.ToTensor()

    train_raw = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    test_raw = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)

    full_dataset = ConcatDataset([train_raw, test_raw])

    n_total = len(full_dataset)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val

    train_set, val_set, test_set = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator = torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(train_set, batch_size = BATCH_SIZE, shuffle = True)
    val_loader = DataLoader(val_set, batch_size = BATCH_SIZE, shuffle = False)
    test_loader = DataLoader(test_set, batch_size = BATCH_SIZE, shuffle = False)

    print(f'Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}')
    return train_loader, val_loader, test_loader

def get_test_tensors(test_loader):

    x_all = []
    y_all = []

    for x, y in test_loader:
        x_all.append(x.view(-1, IMAGE_SIZE))
        y_all.append(y)

    return torch.cat(x_all), torch.cat(y_all) 
