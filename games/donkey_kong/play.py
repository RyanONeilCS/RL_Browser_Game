"""Watch a trained agent play Donkey Kong in a window (real time).

    python games/donkey_kong/play.py

v1, written by Claude on 2026-10-07 (modelled on games/blue_car/play.py).
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))   # models/ paths below are relative to this game's folder

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from donkey_kong_env import DonkeyKongEnv

MODEL = "models/v1_final"  # or a checkpoint, e.g. "models/v1_checkpoints/v1_500000_steps"
EPISODES = 5

# Must be wrapped exactly like in train.py (same frame stacking), or the model sees the wrong shape.
env = VecFrameStack(DummyVecEnv([lambda: DonkeyKongEnv(headless=False)]), n_stack=4)
model = PPO.load(MODEL)

obs = env.reset()
for episode in range(EPISODES):
    done, total, steps = False, 0.0, 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = env.step(action)
        total += reward[0]
        steps += 1
        done = dones[0]
    print(f"episode {episode}: reward {total:.1f}, {steps} steps, best line {infos[0]['height']}")

input("Press Enter to close...")
env.close()
