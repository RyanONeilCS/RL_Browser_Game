"""GameEnv: a Gymnasium env on top of any Adapter.

SKELETON: the plumbing (adapter, config, action mapping) is done. Every TODO is
yours to write: observation space, observations, reward, termination.

Useful references:
  - Gymnasium env API: https://gymnasium.farama.org/introduction/create_custom_env/
  - Raw toy-car state fields: see getState() in games/toy_car/game.js
  - Check your env when done: gymnasium.utils.env_checker.check_env(env)
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from rl_games.adapters import make_adapter
from rl_games.core.config import load_config


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, config: dict | str, obs_mode: str = "state", render_mode: str | None = None):
        super().__init__()
        self.config = load_config(config) if isinstance(config, str) else config
        self.obs_mode = obs_mode              # "state" or "pixels", you decide what each means
        self.render_mode = render_mode

        # Action i -> list of keys to hold, straight from the YAML config.
        self.actions: list[list[str]] = self.config["actions"]
        self.action_space = gym.spaces.Discrete(len(self.actions))

        # TODO: define self.observation_space for each obs_mode.
        #   state:  a Box of your feature vector (size? bounds?)
        #   pixels: a Box of uint8 images (shape? grayscale? frame stack?)
        self.observation_space = None

        self.adapter = make_adapter(self.config)
        self.adapter.start()
        self._last_state: dict[str, Any] | None = None

    # ---- Gymnasium API ---------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        state = self.adapter.reset(seed=seed)
        self._last_state = state

        # TODO: build the first observation (and any info you want to log).
        obs = self._get_obs(state)
        info: dict[str, Any] = {}
        return obs, info

    def step(self, action: int):
        keys = self.actions[int(action)]
        prev = self._last_state
        state = self.adapter.step(keys)
        self._last_state = state

        # TODO: fill these in.
        obs = self._get_obs(state)
        reward = self._compute_reward(prev, state)
        terminated = False   # episode ended because of the game (crash, finish)?
        truncated = False    # episode cut off for another reason (time limit)?
        info: dict[str, Any] = {}
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self.adapter.get_frame()
        return None

    def close(self):
        self.adapter.close()

    # ---- yours to implement ---------------------------------------------
    def _get_obs(self, state: dict[str, Any] | None) -> np.ndarray:
        # TODO: state mode: turn the raw state dict into a normalized feature vector.
        #       pixels mode: self.adapter.get_frame() -> preprocess.
        raise NotImplementedError

    def _compute_reward(self, prev: dict[str, Any] | None, state: dict[str, Any] | None) -> float:
        # TODO: your reward function.
        raise NotImplementedError
