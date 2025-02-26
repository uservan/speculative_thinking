import __init__
import os
import numpy as np
import random
import torch
from utils.setting import *

if project_dir is not None:
    def set_global(path):
        return os.path.join(project_dir, path)

def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def Logger(content):
    print(content)