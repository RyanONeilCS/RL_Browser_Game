"""PPO that keeps learning from human demonstrations while it trains (a "demonstration loss"), for any game.

After every normal PPO update it does a few extra small updates that pull the policy toward the demonstrated
actions (cross-entropy, like behavior cloning), scaled by a weight that shrinks to 0 over `decay_steps`.
At the start the demos guide it into places it rarely reaches; at the end it's pure PPO (learning from reward
only), so it can become better than the demonstrator. The rewards still decide what it learns: with a small
weight, wherever the reward shows something better than the demo, the reward wins.

    model = DemoPPO.load("models/v7b_final", env=env, ...)
    model.set_demos(X, y, weight=0.1, decay_steps=500_000)    # X: stacked observations (N, H, W, C) uint8, y: actions
    model.learn(...)

The demos are not saved inside the model file (they're big); call set_demos again after loading.
"""
import numpy as np
import torch
from stable_baselines3 import PPO


class DemoPPO(PPO):
    demo_steps_per_update = 160      # extra small updates after each PPO update (PPO itself does ~1,600)
    demo_batch = 64

    def set_demos(self, X, y, weight=0.1, decay_steps=500_000):
        """X: (N, H, W, C) uint8 observations exactly as the env gives them (frame-stacked), y: (N,) actions."""
        self._demo_X = torch.as_tensor(X.transpose(0, 3, 1, 2), device=self.device)   # channels first, like training
        self._demo_y = torch.as_tensor(y, device=self.device, dtype=torch.long)
        self._demo_weight0, self._demo_decay = weight, decay_steps
        self._demo_start = self.num_timesteps

    def demo_weight(self):
        if getattr(self, "_demo_X", None) is None:
            return 0.0
        done = (self.num_timesteps - self._demo_start) / max(1, self._demo_decay)
        return self._demo_weight0 * max(0.0, 1.0 - done)

    def train(self):
        super().train()                                   # the normal PPO update (learning from reward)
        w = self.demo_weight()
        self.logger.record("train/demo_weight", w)
        if w <= 0:
            return
        self.policy.set_training_mode(True)
        n, losses, correct = len(self._demo_y), [], 0
        for _ in range(self.demo_steps_per_update):
            idx = torch.randint(0, n, (self.demo_batch,), device=self.device)
            obs, acts = self._demo_X[idx], self._demo_y[idx]
            _, log_prob, _ = self.policy.evaluate_actions(obs, acts)
            loss = -log_prob.mean()                      # how unlikely the demonstrated action is
            self.policy.optimizer.zero_grad()
            (w * loss).backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
            losses.append(loss.item())
        self.logger.record("train/demo_loss", float(np.mean(losses)))   # lower = closer to the demonstrations

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ["_demo_X", "_demo_y"]
