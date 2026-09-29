"""Train an agent on CarEnv. Yours to write."""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))   # models/ and runs/ paths below are relative to this game's folder

from car_env import CarEnv
from core.training_stats import DrivingStats

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback


# TODO: create the env, create PPO, train, save the model

env = CarEnv(difficulty="hard",record_file="runs/hard_v3_run2.jsonl")

eval_env = CarEnv(difficulty="hard", random_starts=True)   # separate game, no record_file, random start each test

eval_cb = EvalCallback(
    eval_env,
    eval_freq=10_000,                  # test every 10k training steps
    n_eval_episodes=5,                 # 5 different starts, so "best" = best driver, not best at one lap
    deterministic=True,
    best_model_save_path="models/hard/v3_run2_best/",
)


model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="runs/")

model.learn(total_timesteps = 750000, callback=[DrivingStats(), eval_cb], tb_log_name="hard_v3_run2")

model.save("models/hard/hard_v3_run2_final")


env.close()
eval_env.close()