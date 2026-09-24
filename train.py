"""Train an agent on CarEnv. Yours to write."""
from car_env import CarEnv

from stable_baselines3 import PPO

# TODO: create the env, create PPO, train, save the model

env = CarEnv(record_file="runs/my_run.jsonl")

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="runs/")

model.learn(total_timesteps = 20000)

model.save("models/car_easy")

env.close()