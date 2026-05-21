"""Central seed management — enforces full determinism before any simulation."""

import os
import random

import numpy as np


def apply_all_seeds(seed: int = 42):
    """Seed Python, NumPy, and PyTorch RNGs plus PYTHONHASHSEED."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def load_seeds(seeds_yaml: str = None) -> dict:
    import yaml
    from pathlib import Path
    if seeds_yaml is None:
        seeds_yaml = Path(__file__).resolve().parents[1] / "config" / "seeds.yaml"
    with open(seeds_yaml) as f:
        return yaml.safe_load(f)
