"""Load per-game YAML configs.

A config describes *how to talk to a game*: where it lives, which adapter to use,
what the discrete actions are (as lists of keys), and timing settings. It says
nothing about rewards or learning; that belongs in your env/agent code.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULTS: dict[str, Any] = {
    "adapter": "browser",
    "timing": "stepped",         # "stepped" (game exposes stepGame) or "realtime"
    "frames_per_action": 4,      # stepped: physics frames per step; realtime: how long to hold keys
    "fps": 60,                   # realtime: game frames per second, used to convert frames -> seconds
    "headless": False,
    "viewport": {"width": 700, "height": 700},
    "canvas_selector": "canvas",
    "frame_selector": None,      # CSS selector of an <iframe> the game lives in (e.g. on itch.io)
    "click_before_start": [],    # selectors to click after loading (e.g. itch's "Run game" button)
    "ready_expr": None,          # JS expression that is truthy once the game is ready
    "state_expr": None,          # JS expression returning a state dict (None: game has no JS state)
    "reset_expr": None,          # JS expression that resets the game, may use `seed`
    "step_expr": None,           # JS expression for stepped mode, may use `keys` and `frames`
    "setup_exprs": [],           # JS statements run once after the game is ready
    "load_timeout_ms": 30000,
}


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = REPO_ROOT / path
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = {**DEFAULTS, **raw}
    cfg["config_path"] = str(path)

    if "url" not in cfg:
        raise ValueError(f"{path}: 'url' is required")
    cfg["url"] = _resolve_url(cfg["url"], path.parent)

    actions = cfg.get("actions")
    if not actions or not all(isinstance(a, list) for a in actions):
        raise ValueError(f"{path}: 'actions' must be a non-empty list of key lists, e.g. [[], [ArrowUp]]")
    if cfg["timing"] not in ("stepped", "realtime"):
        raise ValueError(f"{path}: timing must be 'stepped' or 'realtime'")
    if cfg["timing"] == "stepped" and not cfg["step_expr"]:
        raise ValueError(f"{path}: timing 'stepped' needs a 'step_expr'")
    return cfg


def _resolve_url(url: str, base: Path) -> str:
    """Leave http(s)/file URLs alone; turn a relative file path into a file:// URL.

    The path is tried relative to the config file first, then the repo root.
    Query strings are kept (e.g. ``games/toy_car/index.html?debug``).
    """
    if "://" in url:
        return url
    file_part, sep, query = url.partition("?")
    for root in (base, REPO_ROOT):
        p = (root / file_part).resolve()
        if p.exists():
            return p.as_uri() + (sep + query if sep else "")
    raise FileNotFoundError(f"game file not found: {file_part}")
