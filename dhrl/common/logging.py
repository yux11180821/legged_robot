# Copyright (c) 2024-2026.
# SPDX-License-Identifier: BSD-3-Clause
"""Training-time logging.

  * ``InferenceTimer`` -- single-step policy inference time (推理一步耗时), the
    efficiency metric reported against slow VLM controllers.  On CUDA it
    synchronizes so the timing is real, not async-launch time.
  * ``StepLogger`` -- writes one CSV row + a printed line every ``log_every``
    env-steps; works with vectorized envs (env_steps jumps by num_envs).

Context for this reproduction of arXiv:2407.06499 ("Learning a Distributed
Hierarchical Locomotion Controller for Embodied Cooperation", CoRL 2024). This
module supplies the two metrics the reproduction plan tracks alongside reward
and success rate: (a) the wall-clock cost of ONE forward pass of the
three-layer HRL policy (upper-layer perception -> middle-layer GRU command ->
frozen lower-layer locomotion operator), captured by ``InferenceTimer`` and used
to argue the learned decentralised controller is cheap to run at inference time
(in contrast to heavy VLM-based controllers); and (b) the periodic CSV+stdout
training trace produced by ``StepLogger``, whose per-seed CSVs are later read by
``common.plotting`` to draw the mean +/- std shadow learning curves. Neither
class touches the policy or the environment -- they are pure measurement and
bookkeeping helpers, kept free of any RL logic so they can wrap any of the
single-agent (stage-1) or multi-agent IPPO (stage-2) trainers unchanged.
"""
from __future__ import annotations

import csv
import statistics
import time
from contextlib import contextmanager
from pathlib import Path


class InferenceTimer:
    """Rolling-window timer for the cost of a single policy forward pass.

    Reports the efficiency metric of the reproduction: the milliseconds taken by
    ONE inference step of the agent's three-layer HRL policy (upper-layer
    perception of the exteroceptive observation -> middle-layer GRU command ->
    frozen lower-layer locomotion operator). Because this controller is meant to
    run fully decentralised and on-robot, its per-step latency is a headline
    advantage over slow VLM controllers, so we measure it directly. On CUDA the
    measurement brackets the timed region with ``torch.cuda.synchronize()`` so we
    record true compute time rather than the near-zero async kernel-launch time.
    Only the most recent ``window`` samples are retained, giving a stable, drift-
    free estimate over the course of a long training run.
    """

    def __init__(self, device: str | None = None, window: int = 2000):
        """Set up the sample buffer and decide whether CUDA sync is needed.

        Args:
            device: The torch device string the policy runs on (e.g. ``"cpu"``,
                ``"cuda"``, ``"cuda:0"``), or ``None``. Scalar; only its string
                form is inspected. If it contains ``"cuda"`` and torch reports a
                usable GPU, timings will be CUDA-synchronized; otherwise the
                timer falls back to plain CPU wall-clock timing.
            window: Maximum number of most-recent per-step samples (each a float
                in milliseconds) to keep for the rolling statistics. Scalar int.

        Returns:
            None. Initialises instance state only.
        """
        self._samples_ms: list[float] = []  # rolling buffer of per-step times (ms)
        self._window = window               # cap on buffer length (drops oldest)
        self._cuda = False                  # whether to synchronize() around timing
        self._torch = None                  # lazily-bound torch module (if importable)
        # Only attempt the torch import when the device actually names CUDA; on a
        # CPU-only run we skip it entirely so the timer has no hard torch dependency.
        if device is not None and "cuda" in str(device):
            try:
                import torch

                self._torch = torch
                self._cuda = torch.cuda.is_available()  # guard: CUDA build but no GPU
            except ImportError:
                self._cuda = False  # torch absent -> degrade gracefully to CPU timing

    @contextmanager
    def measure(self):
        """Context manager that times the wrapped block and records one sample.

        Wrap exactly the policy forward pass (the act/inference call of the HRL
        policy) in ``with timer.measure():`` so the recorded sample is the pure
        controller latency reported as the efficiency metric. On CUDA it
        synchronizes both before starting the clock (to flush any pending work
        so it is not charged to this step) and before stopping it (so the GPU
        has actually finished), yielding a faithful compute-time measurement.

        Args:
            (none) -- operates on the wrapped ``with``-block via ``yield``.

        Returns:
            None is yielded; the manager's effect is the side-effect of
            appending one elapsed-time sample (milliseconds, float) to the
            rolling buffer once the block exits. ``finally`` guarantees the
            sample is recorded even if the wrapped block raises.
        """
        if self._cuda:
            self._torch.cuda.synchronize()  # flush pending GPU work BEFORE timing starts
        t0 = time.perf_counter()  # monotonic high-resolution clock for the interval
        try:
            yield  # <-- the caller's policy forward pass runs here
        finally:
            if self._cuda:
                self._torch.cuda.synchronize()  # wait for GPU to finish BEFORE stopping clock
            self._samples_ms.append((time.perf_counter() - t0) * 1000.0)  # s -> ms
            # Keep only the last `window` samples (rolling window over the run).
            if len(self._samples_ms) > self._window:
                self._samples_ms = self._samples_ms[-self._window :]

    def stats(self) -> dict[str, float]:
        """Summarise the rolling buffer into mean / median / p95 latency.

        These three scalars are the inference-time efficiency numbers logged to
        CSV (columns ``infer_ms_*``) and reported for the controller. Median and
        p95 are reported alongside the mean because per-step times are right-
        skewed (occasional scheduling/GC spikes), so the tail (p95) and the
        robust centre (median) characterise the controller better than the mean
        alone.

        Args:
            (none) -- reads the instance's rolling sample buffer.

        Returns:
            A dict[str, float] with keys ``infer_ms_mean``, ``infer_ms_median``,
            ``infer_ms_p95``, each a scalar in milliseconds. When no samples
            have been collected yet, all three are ``NaN`` (so a downstream
            logger/plotter treats them as "no data" rather than as a real 0 ms).
        """
        if not self._samples_ms:
            nan = float("nan")  # no samples yet -> emit NaN sentinels, not zeros
            return {"infer_ms_mean": nan, "infer_ms_median": nan, "infer_ms_p95": nan}
        s = sorted(self._samples_ms)  # ascending copy so we can index the p95 quantile
        return {
            "infer_ms_mean": float(statistics.fmean(self._samples_ms)),
            "infer_ms_median": float(statistics.median(self._samples_ms)),
            # 95th percentile by index; clamp to last element so it never overruns.
            "infer_ms_p95": float(s[min(len(s) - 1, int(0.95 * len(s)))]),
        }


class StepLogger:
    """Throttled CSV + stdout logger keyed on cumulative environment steps.

    Emits one CSV row and one printed line roughly every ``log_every``
    environment steps, producing the per-seed ``*_metrics.csv`` trace that
    ``common.plotting`` later turns into mean +/- std shadow learning curves over
    seeds. Throttling is by a "bucket" of the cumulative ``env_steps`` counter
    rather than by call count, which is essential here because training uses
    VECTORIZED environments: ``env_steps`` advances by ``num_envs`` per update,
    so the logger must fire whenever the step count crosses a new
    ``log_every``-sized bucket, not on every trainer iteration. The logged
    schema covers reward and success rate (task progress on the §5.1
    cooperation tasks), the IPPO/PPO optimisation diagnostics (policy loss,
    value loss, entropy of the clipped-PPO objective), and the
    ``InferenceTimer`` efficiency metrics.
    """

    def __init__(self, csv_path: str | Path, *, log_every: int = 2000,
                 fields: list[str] | None = None, print_fn=print):
        """Configure the CSV target, throttle interval and column schema.

        Args:
            csv_path: Destination path (``str`` or ``Path``) for this run's
                metrics CSV (typically one file per seed, e.g.
                ``<exp>_seed_<k>_metrics.csv``). Parent directories are created.
            log_every: Throttle interval in cumulative environment steps; a row
                is written each time ``env_steps`` enters a new bucket of this
                size. Scalar int, floored at 1 so it can never be zero/negative.
            fields: Ordered list of CSV column names. When ``None``, a default
                schema is used: ``env_steps`` plus task metrics (``reward``,
                ``success_rate``), the three ``infer_ms_*`` efficiency metrics,
                and the PPO diagnostics (``policy_loss``, ``value_loss``,
                ``entropy``). The column order here fixes the CSV header order.
            print_fn: Callable used for the human-readable stdout line (defaults
                to the builtin ``print``); swappable for a logger or a no-op.

        Returns:
            None. Initialises instance state and rotates any stale CSV.
        """
        self.csv_path = Path(csv_path)
        self.log_every = max(1, log_every)   # floor at 1 -> guard against div-by-zero
        self.print_fn = print_fn
        self._last_bucket = -1               # last `env_steps // log_every` we logged
        self._header_written = False         # whether this process has written a header
        self.fields = fields or ["env_steps", "reward", "success_rate",
                                 "infer_ms_mean", "infer_ms_median", "infer_ms_p95",
                                 "policy_loss", "value_loss", "entropy"]
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        # Rotate a pre-existing CSV so a rerun never mixes two runs in one file.
        if self.csv_path.exists() and self.csv_path.stat().st_size > 0:
            self.csv_path.rename(self.csv_path.with_suffix(self.csv_path.suffix + ".prev"))

    def should_log(self, env_steps: int) -> bool:
        """Return whether the step counter has crossed into a new log bucket.

        Encapsulates the throttling decision so callers can branch on it (e.g.
        to compute expensive metrics only when a row is actually due). Bucketing
        on ``env_steps // log_every`` -- rather than on a per-call counter -- is
        what makes throttling correct under vectorized envs, where ``env_steps``
        can jump by ``num_envs`` and skip over several call-count thresholds in
        a single trainer iteration.

        Args:
            env_steps: Cumulative number of environment steps taken so far in
                the run. Scalar int; monotonically non-decreasing across calls.

        Returns:
            ``True`` iff ``env_steps`` falls in a strictly later bucket than the
            last one logged (``self._last_bucket``); ``False`` otherwise. Pure
            predicate -- does NOT advance the bucket (only :meth:`log` does).
        """
        return env_steps // self.log_every > self._last_bucket

    def log(self, env_steps: int, scalars: dict[str, float]) -> None:
        """Unconditionally append one CSV row and print one line for this step.

        Writes a single row of the metrics trace (one point on each per-seed
        learning curve) and advances the throttle bucket. This is the
        force-write path; callers normally reach it via :meth:`maybe_log` so the
        write respects ``log_every``. Missing metrics are written as empty cells
        so the CSV stays rectangular even when a trainer omits, say, the PPO
        diagnostics on a given iteration.

        Args:
            env_steps: Cumulative environment-step count for this row; written
                verbatim into the ``env_steps`` column and used to set the new
                throttle bucket. Scalar int.
            scalars: Mapping from metric name to value for this row (e.g.
                ``reward``, ``success_rate``, ``policy_loss``, ``value_loss``,
                ``entropy``, and the ``infer_ms_*`` keys from
                ``InferenceTimer.stats``). Keys outside ``self.fields`` are
                ignored; fields absent from this mapping become empty cells.

        Returns:
            None. Side effects only: one appended CSV row plus one printed line.
        """
        self._last_bucket = env_steps // self.log_every  # mark this bucket as logged
        # Project scalars onto the fixed column schema; absent metrics -> "" (blank cell).
        row = {k: scalars.get(k, "") for k in self.fields}
        row["env_steps"] = env_steps  # always stamp the step counter, even if not in `scalars`
        with self.csv_path.open("a", newline="") as fh:  # append mode -> never truncates prior rows
            w = csv.DictWriter(fh, fieldnames=self.fields)
            # Write the header exactly once, and only when the file is genuinely empty
            # (so an in-process re-entry after rotation still gets a header).
            if not self._header_written and self.csv_path.stat().st_size == 0:
                w.writeheader()
            w.writerow(row)
        self._header_written = True
        # Build the stdout line: floats get 4 decimals; everything else prints as-is.
        msg = " ".join(
            f"{k}={scalars[k]:.4f}" if isinstance(scalars.get(k), float) else f"{k}={scalars.get(k)}"
            for k in self.fields if k in scalars  # only print metrics actually supplied
        )
        self.print_fn(f"[step {env_steps}] {msg}")

    def maybe_log(self, env_steps: int, scalars: dict[str, float]) -> bool:
        """Log this step iff the throttle bucket has advanced; report if it did.

        The convenience method trainers call every iteration: it folds the
        :meth:`should_log` check and the :meth:`log` write into one call so the
        training loop need not track buckets itself. Returning whether a row was
        actually written lets the caller skip recomputing expensive summaries on
        the (common) no-op iterations.

        Args:
            env_steps: Cumulative environment-step count for this iteration.
                Scalar int; forwarded to both the throttle check and the write.
            scalars: Per-step metric mapping, same schema as :meth:`log`.

        Returns:
            ``True`` if a CSV row + stdout line were emitted on this call;
            ``False`` if the call was throttled (no new bucket) and nothing
            was written.
        """
        if self.should_log(env_steps):  # only write on a fresh bucket
            self.log(env_steps, scalars)
            return True
        return False
