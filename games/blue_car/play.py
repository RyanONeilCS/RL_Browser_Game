"""Watch a trained agent drive Blue Car in a window. Framework only: needs a trained model."""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))   # models/ paths below are relative to this game's folder

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from blue_car_env import BlueCarEnv

MODEL = "models/v1_final"       # or a checkpoint, e.g. "models/v1_checkpoints/v1_500000_steps"
EPISODES = 3

# Must be wrapped exactly like in train.py (same frame stacking), or the model sees the wrong shape.
env = VecFrameStack(DummyVecEnv([lambda: BlueCarEnv(headless=False)]), n_stack=4)
model = PPO.load(MODEL)

# The game already runs in real time, so no sleep is needed (unlike the toy car's stepped mode).
# A VecEnv resets itself when an episode ends and returns lists (one entry per env).
for episode in range(EPISODES):
    obs = env.reset()
    done = False
    total, steps = 0.0, 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = env.step(action)
        total += reward[0]
        steps += 1
        done = dones[0]
    print(f"episode {episode}: reward {total:.2f}, {steps} steps")

input("Press Enter to close...")
env.close()
