"""Extra driving stats in SB3's terminal output (and TensorBoard), under rollout/:

    avg_speed   average speed over each episode, as a fraction of the top speed (1.0 = flat out)
    lap_rate    fraction of episodes that finished the lap
    lap_time    average lap time in seconds, finished laps only

Reads "speed", "seconds" and "lap" from the env's info dict; games whose info doesn't have them
are skipped (no stats, no error). Use: model.learn(..., callback=DrivingStats())
"""
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class DrivingStats(BaseCallback):
    def _on_training_start(self):
        n = self.training_env.num_envs
        self._speed_sum = np.zeros(n)
        self._steps = np.zeros(n)

    def _on_rollout_start(self):
        # Stats of episodes that ended during this rollout (episodes can span rollouts).
        self._ep_speeds, self._laps, self._lap_times = [], [], []

    def _on_step(self):
        for i, (info, done) in enumerate(zip(self.locals["infos"], self.locals["dones"])):
            if not {"speed", "seconds", "lap"} <= info.keys():
                continue   # not a racing game
            self._speed_sum[i] += info["speed"]
            self._steps[i] += 1
            if done:
                self._ep_speeds.append(self._speed_sum[i] / self._steps[i])
                finished = info["lap"] >= 1
                self._laps.append(finished)
                if finished:
                    self._lap_times.append(info["seconds"])
                self._speed_sum[i] = self._steps[i] = 0
        return True

    def _on_rollout_end(self):
        if self._ep_speeds:
            self.logger.record("rollout/avg_speed", np.mean(self._ep_speeds))
            self.logger.record("rollout/lap_rate", np.mean(self._laps))
        if self._lap_times:
            self.logger.record("rollout/lap_time", np.mean(self._lap_times))
