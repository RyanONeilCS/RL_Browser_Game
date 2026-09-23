"""The interface every game backend implements.

An adapter only moves inputs and outputs between Python and a game: it presses
keys, grabs frames, and reads raw state. It does not know about observations,
rewards, or learning. New backends (desktop screen capture, Unity ML-Agents, ...)
subclass this so that the same env and agent code works on top of them.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np


class Adapter(ABC):
    def __init__(self, config: dict[str, Any]):
        self.config = config

    @abstractmethod
    def start(self) -> None:
        """Launch/connect to the game. Call once before anything else."""

    @abstractmethod
    def reset(self, seed: int | None = None) -> dict[str, Any] | None:
        """Start a new episode. Returns the raw game state if the game exposes one."""

    @abstractmethod
    def step(self, keys: Sequence[str], frames: int | None = None) -> dict[str, Any] | None:
        """Apply `keys` for `frames` game frames (config default if None).

        Stepped games advance exactly that many frames. Real-time games get the keys
        held for about that long in wall-clock time. Returns the raw state (or None).
        """

    @abstractmethod
    def get_state(self) -> dict[str, Any] | None:
        """Raw game state as a dict, or None if the game has no readable state."""

    @abstractmethod
    def get_frame(self) -> np.ndarray:
        """Current game image as an RGB uint8 array of shape (H, W, 3)."""

    @abstractmethod
    def close(self) -> None:
        """Release everything (browser, processes, windows)."""

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.close()
