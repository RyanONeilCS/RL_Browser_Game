"""Test a model (or a random agent) on Blue Car without a window: reward and length per episode.

MODEL = None tests a random agent: use that for checklist step 9 (do episodes end, does reset work,
do the rewards look sensible?) before any training, and as the baseline to beat afterwards.
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))   # models/ paths below are relative to this game's folder

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from blue_car_env import BlueCarEnv

MODEL = None               # e.g. "models/v1_final" or "models/v1_checkpoints/v1_500000_steps"
EPISODES = 5

env = VecFrameStack(DummyVecEnv([lambda: BlueCarEnv()]), n_stack=4)   # same wrapping as train.py
model = PPO.load(MODEL) if MODEL else None

results = []
for episode in range(EPISODES):
    obs = env.reset()
    done = False
    total, steps = 0.0, 0
    while not done:
        if model:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = [env.action_space.sample()]
        obs, reward, dones, infos = env.step(action)
        total += reward[0]
        steps += 1
        done = dones[0]
    results.append((total, steps))
    # TODO: print useful values from infos[0] too (score, why it ended, ...)
    print(f"episode {episode}: reward {total:7.2f}  steps {steps}")

print(f"\naverage reward {sum(r for r, _ in results) / EPISODES:.2f}, "
      f"average length {sum(s for _, s in results) / EPISODES:.0f} steps")
env.close()
