"""Minimal random-state helpers; no host, user or environment information saved."""

import random
import numpy as np
import torch


def seed_everything(seed=42, deterministic=False):
    """Seed training RNGs; exact reproducibility across platforms is not assured.

    Strict deterministic mode may reject unsupported operations. This function
    does not mutate process environment variables or set PYTHONHASHSEED at runtime.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """DataLoader worker_init_fn; uses the worker seed assigned by PyTorch."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
