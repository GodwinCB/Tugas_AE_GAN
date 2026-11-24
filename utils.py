# Helper kecil -> seed, simpan/muat checkpoint, simpan history, simpan figur

import os
import json
import numpy as np
import torch

from config import SEED, DEVICE, CHECKPOINT_DIR, RESULT_DIR, FIGURE_DIR

def set_seed(seed = SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)

def ensure_dirs():
    for d in (CHECKPOINT_DIR, RESULT_DIR, FIGURE_DIR):
        os.makedirs(d, exist_ok = True)

def save_checkpoint(model, name):
    torch.save(model.state_dict(), f'{CHECKPOINT_DIR}/{name}.pt')

def load_checkpoint(model, name):
    model.load_state_dict(torch.load(f'{CHECKPOINT_DIR}/{name}.pt', map_location = DEVICE))
    return model.to(DEVICE).eval()

def save_json(name, data):
    with open(f'{RESULT_DIR}/{name}.json', 'w') as f:
        json.dump(data, f, indent = 2)

def load_json(name):
    with open(f'{RESULT_DIR}/{name}.json') as f:
        return json.load(f)

def save_fig(name):
    import matplotlib.pyplot as plt
    plt.savefig(f'{FIGURE_DIR}/{name}.png', dpi = 200, bbox_inches = 'tight')
    plt.close()