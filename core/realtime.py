"""Helpers for training on real-time games (games that keep running while Python isn't looking).

    model.learn(..., callback=[RestartAfterUpdate(), InfoStats(["cubes"])])
"""
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class RestartAfterUpdate(BaseCallback):
    """PPO stops collecting to update the network (seconds), but a real-time game keeps running with the
    last keys still held, so every car would drive blind. This releases all keys when a batch is collected
    and restarts every game before the next batch, so each batch starts from a clean state.

    The env must have a release_keys() method. Episodes are cut at batch boundaries, so keep an
    episode's time limit about as long as one batch (n_steps per game)."""

    def _on_rollout_end(self):
        self.training_env.env_method("release_keys")

    def _on_rollout_start(self):
        if self.num_timesteps == 0:
            return   # the first batch: learn() has just reset every game
        self.model._last_obs = self.training_env.reset()
        self.model._last_episode_starts = np.ones(self.training_env.num_envs, dtype=bool)

    def _on_step(self):
        return True


class InfoStats(BaseCallback):
    """Logs the average of info[key] at the end of each episode (e.g. cubes collected) as rollout/<key>,
    in the terminal output and TensorBoard."""

    def __init__(self, keys):
        super().__init__()
        self.keys = keys

    def _on_rollout_start(self):
        self._ended = {k: [] for k in self.keys}

    def _on_step(self):
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done:
                for k in self.keys:
                    if k in info:
                        self._ended[k].append(info[k])
        return True

    def _on_rollout_end(self):
        for k, values in self._ended.items():
            if values:
                self.logger.record(f"rollout/{k}", float(np.mean(values)))
