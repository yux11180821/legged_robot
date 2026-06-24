# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Reproducible seeding across random / numpy / torch (+ CUDA).

Self-contained (no matplotlib/util import) so the training entry point can seed
without pulling in the plotting stack.  Functionally matches ``util.set_seed``.
"""
from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
