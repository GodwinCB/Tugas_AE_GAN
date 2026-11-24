# Tahap 1: MLP Classifier (784 -> 256 -> 64 -> 10)

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DEVICE, IMAGE_SIZE, HIDDEN_DIM_1, LATENT_DIM, NUM_CLASSES, MAX_EPOCHS, LEARNING_RATE
from data import get_loaders
from utils import set_seed, ensure_dirs, save_checkpoint, save_json

class MLPClassifier(nn.Module):
    def __init__(self):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(IMAGE_SIZE, HIDDEN_DIM_1)
        self.fc2 = nn.Linear(HIDDEN_DIM_1, LATENT_DIM)
        self.fc3 = nn.Linear(LATENT_DIM, NUM_CLASSES)

    # Ambil fitur laten z_MLP
    def features(self, x):
        h1 = F.relu(self.fc1(x))
        z = F.relu(self.fc2(h1))
        return z

    def forward(self, x):
        z = self.features(x)
        return self.fc3(z) # Logits

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.view(-1, IMAGE_SIZE).to(DEVICE)
            y = y.to(DEVICE)
            pred = model(x).argmax(dim = 1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

def main():
    ensure_dirs()
    set_seed()
    train_loader, val_loader, test_loader = get_loaders()

    clf = MLPClassifier().to(DEVICE)
    print(clf)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(clf.parameters(), lr = LEARNING_RATE)

    loss_history = []
    val_acc_history = []
    best_val_acc = 0.0
    best_state = None

    for epoch in range(MAX_EPOCHS):
        clf.train()
        epoch_loss = []

        for x, y in train_loader:
            # Flatten batch image
            x = x.view(-1, IMAGE_SIZE).to(DEVICE)
            y = y.to(DEVICE)

            logits = clf(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())

        mean_loss = np.mean(epoch_loss)
        val_acc = evaluate(clf, val_loader)
        loss_history.append(mean_loss)
        val_acc_history.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(clf.state_dict())

        print(f'Epoch[{epoch + 1} / {MAX_EPOCHS}], Train Loss: {mean_loss:.4f}, Val Acc: {val_acc:.4f}')

    clf.load_state_dict(best_state)
    test_acc = evaluate(clf, test_loader)
    print(f'Test Accuracy: {test_acc:.4f}')

    assert test_acc >= 0.88, 'Akurasi test belum mencapai 88%, tambah epoch / ubah hyperparameter'

    save_checkpoint(clf, 'mlp_classifier')
    save_json('mlp_classifier', {'loss': loss_history, 'val_acc': val_acc_history, 'test_acc': test_acc})

if __name__ == '__main__':
    main()