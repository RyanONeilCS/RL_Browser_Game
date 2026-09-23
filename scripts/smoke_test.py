"""Check that the browser plumbing works: random actions, no learning.

    python scripts/smoke_test.py                      # toy car, stepped then realtime, visible browser
    python scripts/smoke_test.py --headless --steps 200
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_games.adapters import make_adapter  # noqa: E402
from rl_games.core.config import REPO_ROOT, load_config  # noqa: E402


def run(cfg: dict, steps: int, out: Path) -> None:
    print(f"\n=== {cfg['game']} | timing={cfg['timing']} | headless={cfg['headless']} ===")
    with make_adapter(cfg) as adapter:
        if cfg["timing"] == "realtime" and cfg["state_expr"]:
            adapter.eval_js("window.setMode('realtime')")
        state = adapter.reset(seed=0)
        episodes, t0 = 1, time.perf_counter()
        for i in range(steps):
            state = adapter.step(random.choice(cfg["actions"]))
            if state and (state.get("done") or state.get("truncated")):
                print(f"  step {i:4d}: episode over (crashed={state['crashed']}, "
                      f"checkpoints={state['checkpointsPassed']}, frame={state['frame']})")
                state = adapter.reset(seed=episodes)
                episodes += 1
        dt = time.perf_counter() - t0
        print(f"  {steps} steps in {dt:.2f}s -> {steps / dt:.0f} steps/s, {episodes} episodes")

        frame = adapter.get_frame()
        path = out / f"smoke_{cfg['game']}_{cfg['timing']}.png"
        cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"  frame {frame.shape} {frame.dtype} saved to {path}")
        if state:
            print("  state keys:", sorted(state))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/toy_car.yaml")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--only", choices=["stepped", "realtime"])
    args = ap.parse_args()

    out = REPO_ROOT / "runs"
    out.mkdir(exist_ok=True)
    base = load_config(args.config)
    base["headless"] = args.headless or base["headless"]

    modes = [args.only] if args.only else ["stepped", "realtime"]
    for mode in modes:
        cfg = dict(base, timing=mode)
        # Real time runs at wall-clock speed (~15 steps/s), so keep it short.
        run(cfg, min(args.steps, 150) if mode == "realtime" else args.steps, out)


if __name__ == "__main__":
    main()
