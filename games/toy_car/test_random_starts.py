"""Did the agent learn to drive, or memorize one path? Drive from random start positions and count laps."""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))   # models/ and runs/ paths below are relative to this game's folder

from car_env import CarEnv
from stable_baselines3 import PPO

MODEL = "models/hard/hard_v1_final"
DIFFICULTY = "hard"      # "easy" or "hard"
EPISODES = 20

env = CarEnv(difficulty=DIFFICULTY)   # headless: we only want the numbers
model = PPO.load(MODEL)

laps = 0
for seed in range(EPISODES):
    obs, _ = env.reset(seed=seed)   # each seed = a different start checkpoint and heading
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    result = "lap" if info["lap"] >= 1 else "CRASH" if info["crashed"] else "time up"
    print(f"seed {seed:2}: {result:7} {info['seconds']:5.2f}s  checkpoints {info['checkpoints']}")
    laps += info["lap"] >= 1

print(f"\n{laps}/{EPISODES} laps finished")
env.close()
