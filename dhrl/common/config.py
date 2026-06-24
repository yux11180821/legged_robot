# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""YAML config loading.

A config may declare ``defaults: [base]`` to inherit from sibling YAML files
(e.g. ``base.yaml``); those are deep-merged first, then the config's own keys
override.  The result is a plain nested dict consumed by ``run.py``.

Context for this reproduction of arXiv:2407.06499 ("Learning a Distributed
Hierarchical Locomotion Controller for Embodied Cooperation", CoRL 2024): the
hyper-parameters of the three-layer HRL policy (upper-layer perception of the
exteroceptive observation, middle-layer GRU memory, frozen lower-layer
locomotion operator) and of the IPPO optimiser (fully decentralised: one
clipped-PPO objective per agent, no centralized critic) live in YAML rather
than in code so that every cooperation task (§5.1) and every robot count can be
declared as a thin override on a shared ``base.yaml``. This module is therefore
the config-loading helper only -- it holds NO policy logic, just the
defaults-inheritance and deep-merge machinery that assembles a run's settings.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _deep_merge(a: dict, b: dict) -> dict:
    """Recursively merge two nested config dicts so ``b`` overrides ``a``.

    This is the primitive behind the ``defaults: [...]`` inheritance mechanism:
    the experiment configs in this reproduction (one per cooperation task of
    §5.1 -- Cooperative Transport / Corridor Crossing / Ravine Bridging -- and
    one per robot count, e.g. the 2/3-Spot variants) all share a common
    ``base.yaml`` of IPPO/PPO hyper-parameters and only override the few keys
    that differ.  Deep-merging (rather than a flat ``dict.update``) lets a child
    config override, say, ``train.lr`` without having to restate the entire
    ``train`` block, which keeps the configs DRY and auditable for the paper's
    two-stage training recipe (§5.2).

    Args:
        a: Base mapping (the "lower-priority" config); shape is an arbitrary
            nested dict[str, Any]. Its keys are kept unless overridden by ``b``.
        b: Override mapping (the "higher-priority" config); same nested
            dict[str, Any] shape. Scalar/list values here win outright; dict
            values are merged recursively into the matching key of ``a``.

    Returns:
        A brand-new nested dict[str, Any] holding the merged result. Neither
        ``a`` nor ``b`` is mutated (we copy ``a`` first), so callers can reuse
        the inputs safely across multiple merges.
    """
    out = dict(a)  # shallow copy so the caller's `a` is never mutated in place
    for k, v in b.items():
        # Only recurse when BOTH sides are dicts; otherwise b's value replaces a's.
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)  # nested block -> merge field-by-field
        else:
            out[k] = v  # scalar / list / new key -> b wins outright
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    """Load one experiment YAML and resolve its ``defaults`` inheritance chain.

    This is the single entry point ``run.py`` uses to materialise the full
    hyper-parameter dict for a run. The returned dict drives both stages of the
    paper's recipe (§5.2): stage-1 pre-training of the FROZEN lower-layer
    locomotion operator, and stage-2 IPPO training of the upper+middle layers
    over the frozen lower layer. Resolution order is "defaults first, own keys
    last", so a task/robot-count config always wins over the shared base it
    inherits from.

    Args:
        path: Filesystem path (``str`` or ``Path``) to the leaf experiment YAML
            for this run. Its optional ``defaults: [name, ...]`` list names
            sibling files (``<name>.yaml`` in the SAME directory) to inherit.

    Returns:
        A plain nested dict[str, Any] with the inheritance fully flattened: each
        listed default is loaded and deep-merged in declaration order, then the
        leaf config's own keys are merged on top (so the leaf overrides its
        bases). No ``defaults`` key remains in the result. This is consumed
        directly by the trainer; there is no schema object, just nested dicts.
    """
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}  # `or {}` -> an empty file parses to {}
    merged: dict[str, Any] = {}
    # Pop `defaults` so it never leaks into the final config; iterate in order
    # so later defaults override earlier ones (standard inheritance semantics).
    for name in cfg.pop("defaults", []):
        # Sibling lookup: defaults are resolved relative to THIS file's folder.
        base = yaml.safe_load((path.parent / f"{name}.yaml").read_text()) or {}
        merged = _deep_merge(merged, base)
    # The leaf config's own keys are merged LAST so they take final precedence.
    return _deep_merge(merged, cfg)
