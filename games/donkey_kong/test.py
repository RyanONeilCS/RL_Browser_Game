"""Test a model (or a random agent) on Donkey Kong without a window: how each episode ends, height reached, reward.

    python games/donkey_kong/test.py

MODEL = None tests a random agent: the baseline to beat (random play: best line ~1, i.e. a jump on the bottom girder).
v1, written by Claude on 2026-10-07 (modelled on games/blue_car/test.py).
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))   # models/ paths below are relative to this game's folder

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from donkey_kong_env import DonkeyKongEnv

MODEL = None               # e.g. "models/v1_final" or "models/v1_checkpoints/v1_500000_steps"
EPISODES = 10
DETERMINISTIC = True       # True: always the most likely action; False: sample like during training
RECORD = True              # save every episode to runs/test_<model>.jsonl

name = os.path.basename(MODEL) if MODEL else "random"
record = f"runs/test_{name}.jsonl" if RECORD else None
env = VecFrameStack(DummyVecEnv([lambda: DonkeyKongEnv(record_file=record, record_every=1)]), n_stack=4)   # as train.py
model = PPO.load(MODEL) if MODEL else None

results = []
obs = env.reset()
for episode in range(EPISODES):
    done, total, steps = False, 0.0, 0
    while not done:
        action = model.predict(obs, deterministic=DETERMINISTIC)[0] if model else [env.action_space.sample()]
        obs, reward, dones, infos = env.step(action)
        total += reward[0]
        steps += 1
        done = dones[0]
    info = infos[0]
    ended = "WON" if info["won"] else "died" if info["died"] else "time limit"
    results.append((total, steps, info["height"], ended))
    print(f"episode {episode}: {ended:10} after {steps:4} steps, best line {info['height']:2}, reward {total:6.1f}")

print(f"\nwon {sum(r[3] == 'WON' for r in results)}/{EPISODES}, "
      f"average best line {sum(r[2] for r in results) / EPISODES:.1f}, "
      f"average reward {sum(r[0] for r in results) / EPISODES:.1f}")
env.close()
