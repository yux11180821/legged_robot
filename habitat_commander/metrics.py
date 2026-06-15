# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Metrics harness required by the advisor's meeting (decomposition-agnostic).

Two hard requirements from the meeting:
  * print a data point every N env-steps (default 20) -> smooth reward curves;
  * record "single-step inference time" (推理一步耗时) -> the efficiency metric
    used to rebut reviewers (our selling point vs a slow VLM controller).

Pure (torch import is optional, only for CUDA-accurate timing).  Unit-tested in
selftest.py.
"""

from __future__ import annotations

import csv
import statistics
import time
from contextlib import contextmanager
from pathlib import Path


class InferenceTimer:
    """Times a single policy forward pass (the '单步推理耗时' metric).

    Use ``with timer.measure(): action = policy.act(...)``.  On CUDA we
    synchronize around the region so the timing is real, not async-launch time.
    """

    def __init__(self, device: str | None = None, window: int = 2000):
        self._samples_ms: list[float] = []
        self._window = window
        self._cuda = False
        self._torch = None
        if device is not None and "cuda" in str(device):
            try:
                import torch
                self._torch = torch
                self._cuda = torch.cuda.is_available()
            except Exception:
                self._cuda = False

    @contextmanager
    def measure(self):
        if self._cuda:
            self._torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self._cuda:
                self._torch.cuda.synchronize()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self._samples_ms.append(dt_ms)
            if len(self._samples_ms) > self._window:
                self._samples_ms = self._samples_ms[-self._window :]

    def stats(self) -> dict[str, float]:
        if not self._samples_ms:
            return {"infer_ms_mean": float("nan"), "infer_ms_median": float("nan"),
                    "infer_ms_p95": float("nan"), "infer_ms_last": float("nan")}
        s = sorted(self._samples_ms)
        p95 = s[min(len(s) - 1, int(0.95 * len(s)))]
        return {
            "infer_ms_mean": float(statistics.fmean(self._samples_ms)),
            "infer_ms_median": float(statistics.median(self._samples_ms)),
            "infer_ms_p95": float(p95),
            "infer_ms_last": float(self._samples_ms[-1]),
        }


class StepLogger:
    """Emit one CSV row + a printed line every ``log_every`` env-steps.

    Works with vectorized envs: ``env_steps`` jumps by ``num_envs`` per update, so
    we log whenever a new ``log_every`` bucket is crossed (no double / skipped rows).
    """

    def __init__(
        self,
        csv_path: str | Path,
        *,
        log_every: int = 20,
        fields: list[str] | None = None,
        print_fn=print,
    ):
        self.csv_path = Path(csv_path)
        self.log_every = max(1, log_every)
        self.print_fn = print_fn
        self._last_bucket = -1
        self._header_written = False
        self.fields = fields or [
            "env_steps", "reward", "episode_return",
            "infer_ms_mean", "infer_ms_median", "infer_ms_p95", "infer_ms_last",
            "policy_loss", "value_loss", "entropy",
        ]
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        # Rotate a pre-existing CSV so a rerun never mixes two runs in one file.
        if self.csv_path.exists() and self.csv_path.stat().st_size > 0:
            self.csv_path.rename(self.csv_path.with_suffix(self.csv_path.suffix + ".prev"))

    def should_log(self, env_steps: int) -> bool:
        return env_steps // self.log_every > self._last_bucket

    def log(self, env_steps: int, scalars: dict[str, float]) -> None:
        self._last_bucket = env_steps // self.log_every
        row = {k: scalars.get(k, "") for k in self.fields}
        row["env_steps"] = env_steps
        with self.csv_path.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=self.fields)
            if not self._header_written and self.csv_path.stat().st_size == 0:
                w.writeheader()
            w.writerow(row)
        self._header_written = True
        msg = " ".join(
            f"{k}={scalars[k]:.4f}" if isinstance(scalars.get(k), float) else f"{k}={scalars.get(k)}"
            for k in self.fields if k in scalars
        )
        self.print_fn(f"[step {env_steps}] {msg}")

    def maybe_log(self, env_steps: int, scalars: dict[str, float]) -> bool:
        if self.should_log(env_steps):
            self.log(env_steps, scalars)
            return True
        return False
