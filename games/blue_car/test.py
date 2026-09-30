"""Test a model (or a random agent) on Blue Car without a window: how each episode ends, cubes, reward.

MODEL = None tests a random agent: use that for checklist step 9 (do episodes end, does reset work,
do the rewards look sensible?) before any training, and as the baseline to beat afterwards.
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))   # models/ paths below are relative to this game's folder

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from blue_car_env import BlueCarEnv

MODEL = None               # e.g. "models/bc_v1", "models/v6_final" or "models/v6_checkpoints/v6_500000_steps"
EPISODES = 5
DETERMINISTIC = True       # True: always the most likely action; False: sample like during training
RECORD = True              # save every episode to runs/test_<model>.jsonl, to watch in viewer.html

name = os.path.basename(MODEL) if MODEL else "random"
record = f"runs/test_{name}.jsonl" if RECORD else None
env = VecFrameStack(DummyVecEnv([lambda: BlueCarEnv(record_file=record, record_every=1)]), n_stack=4)   # same wrapping as train.py
model = PPO.load(MODEL) if MODEL else None

results = []
for episode in range(EPISODES):
    obs = env.reset()
    done = False
    total, steps = 0.0, 0
    while not done:
        if model:
            action, _ = model.predict(obs, deterministic=DETERMINISTIC)
        else:
            action = [env.action_space.sample()]
        obs, reward, dones, infos = env.step(action)
        total += reward[0]
        steps += 1
        done = dones[0]
    info = infos[0]
    ended = ("WON" if info.get("won") else "lost" if info.get("lost") else "off road" if info.get("off_road")
             else "time limit")
    results.append((total, steps, info.get("cubes", 0), ended))
    print(f"episode {episode}: {ended:10} after {steps:4} steps (~{steps * 0.16:.0f} s), "
          f"cubes {info.get('cubes', 0):2}, reward {total:7.2f}")

won = sum(r[3] == "WON" for r in results)
print(f"\nwon {won}/{EPISODES}, average cubes {sum(r[2] for r in results) / EPISODES:.1f}, "
      f"average reward {sum(r[0] for r in results) / EPISODES:.2f}, "
      f"average length {sum(r[1] for r in results) / EPISODES:.0f} steps")
env.close()
if record:
    print(f"episodes saved to games/blue_car/{record}: open viewer.html and choose that file")
