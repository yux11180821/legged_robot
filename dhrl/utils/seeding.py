# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Reproducible seeding across random / numpy / torch (+ CUDA).

Self-contained (no matplotlib/util import) so the training entry point can seed
without pulling in the plotting stack.  Functionally matches ``util.set_seed``.

Context for this reproduction of arXiv:2407.06499 ("Learning a Distributed
Hierarchical Locomotion Controller for Embodied Cooperation", CoRL 2024):
because results are reported as mean +/- std learning curves over multiple seeds
(cf. the across-seed variance bands of Fig.5, drawn via ``common.plotting``),
every run must seed ALL stochastic sources -- Python ``random``, NumPy, and
torch (CPU + every CUDA device) -- from a single integer so that one seed
reproducibly fixes the entire stage-1 (frozen lower-layer pre-training) and
stage-2 (IPPO upper+middle-layer training) pipeline. This module is kept import-
light (no matplotlib/util) precisely so the training entry point can seed before
any heavy module loads.
"""
from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    """Seed every RNG (Python / NumPy / torch CPU+CUDA) from one integer.

    The single point of randomness control for a run: calling this once at the
    start fixes all stochastic sources -- environment resets and sampling
    (NumPy / Python ``random``), network init, and the IPPO policy's action
    sampling from the middle-layer command distribution (torch) -- so that a
    given seed yields a reproducible trajectory through both training stages.
    This is what makes the per-seed CSV traces (and thus the mean +/- std shadow
    curves over seeds) meaningful and re-runnable.

    Args:
        seed: The master seed; one scalar int that is fed identically to
            ``random``, ``numpy``, and torch (CPU + all CUDA devices). Different
            integers give the independent runs that the variance band averages
            over; the same integer reproduces a run exactly.
        deterministic: Keyword-only flag. When ``True``, also forces cuDNN into
            deterministic mode (``cudnn.deterministic = True``,
            ``cudnn.benchmark = False``) so GPU convolution kernels are
            bit-reproducible, at some throughput cost. Left ``False`` by default
            so normal training keeps cuDNN's faster autotuned kernels.

    Returns:
        None. Acts purely by side-effect on the global RNG state of each library.
    """
    random.seed(seed)      # Python's built-in RNG (used by some env/sampling code)
    np.random.seed(seed)   # NumPy global RNG (env resets, noise, shuffles)
    try:
        import torch

        torch.manual_seed(seed)          # seeds the torch CPU RNG
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)  # seed EVERY visible CUDA device's RNG
        if deterministic:
            # Trade speed for bit-exact reproducibility of cuDNN kernels.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # torch optional: on a torch-less install, seed only random + numpy
