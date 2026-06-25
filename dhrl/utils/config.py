# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Config loading (YAML -> argparse.Namespace), matching the MARL_for_MeltingPot style.

A run is fully described by one flat YAML file under ``configs/``; we load it into an
``args`` Namespace that every module reads via ``args.xxx``.  ``load_algorithm`` lets the
config pick the learner by a dotted string (e.g. ``algo: algorithms.ippo.IPPO``).
"""
import argparse
import importlib
import os

import torch
import yaml


def get_config():
    """Parse ``--config <yaml>`` (+ optional ``--seed``) and return an ``args`` Namespace.

    The YAML's keys become attributes of ``args`` (flat, accessed as ``args.lr`` etc.).
    Device is downgraded to CPU automatically if CUDA is requested but unavailable.

    Returns:
        argparse.Namespace with every YAML key as an attribute, plus ``config_file``.
    """
    parser = argparse.ArgumentParser(description="Distributed HRL (IPPO) reproduction.")
    parser.add_argument("--config", type=str, required=True, help="path to a YAML config")
    parser.add_argument("--seed", type=int, default=None, help="override the config seed")
    cli, _ = parser.parse_known_args()

    if not os.path.exists(cli.config):
        raise FileNotFoundError(f"config file not found: {cli.config}")
    with open(cli.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    args = argparse.Namespace(**cfg)
    args.config_file = cli.config
    if cli.seed is not None:
        args.seed = cli.seed
    if not hasattr(args, "seed"):
        args.seed = getattr(args, "seeds", [100])[0]

    if getattr(args, "device", "cpu") == "cuda" and not torch.cuda.is_available():
        print("[config] CUDA unavailable -> using CPU")
        args.device = "cpu"
    return args


def load_algorithm(algo_path):
    """Import a learner class from a dotted path, e.g. ``"algorithms.ippo.IPPO"``.

    Lets the YAML select the algorithm (``algo: ...``) without an if/else in ``run.py``.

    Args:
        algo_path: dotted ``module.ClassName`` string.
    Returns:
        the class object.
    """
    module_name, class_name = algo_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)
