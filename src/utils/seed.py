import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False, benchmark: bool = False) -> None:
    """Set random seed for reproducibility.

    Args:
        seed: seed value.
        deterministic: if True, sets cudnn to deterministic (may slow down).
        benchmark: if True, enables cudnn benchmark (may speed up, less reproducible).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = benchmark
