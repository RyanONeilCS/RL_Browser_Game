"""Watch a trained agent drive. Yours to write."""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))   # models/ and runs/ paths below are relative to this game's folder

import time

from car_env import CarEnv
from stable_baselines3 import PPO

# Which model to watch, and on which track. Mix them to test transfer (e.g. the easy model on "hard").
MODEL = "models/hard/hard_v3_final"
DIFFICULTY = "hard"      # "easy" or "hard"

env = CarEnv(headless=False, difficulty=DIFFICULTY)

# 2. load the trained model
model = PPO.load(MODEL)


for i in range(3):
    obs, _ = env.reset()
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        time.sleep(4 / 60)
        done = terminated or truncated

    print(info["seconds"], info["lap"], info["crashed"])

input("Press Enter to Close")
env.close()