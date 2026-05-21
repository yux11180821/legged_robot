#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[100, 200, 300, 400])
    args = parser.parse_args()

    root = Path(args.checkpoint_root)
    for seed in args.seeds:
        seed_dir = root / f"seed_{seed}"
        resume = seed_dir / ".habitat-resume-state.pth"
        out = seed_dir / "final_from_resume.pth"
        obj = torch.load(resume, map_location="cpu", weights_only=False)
        step = int(obj.get("requeue_stats", {}).get("num_steps_done", 0))
        ckpt = {
            "state_dict": obj["state_dict"],
            "config": obj["config"],
            "extra_state": {"step": step, "source": str(resume)},
        }
        torch.save(ckpt, out)
        print(seed, step, out)


if __name__ == "__main__":
    main()
