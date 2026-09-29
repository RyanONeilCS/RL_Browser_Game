"""Train an agent on Blue Car, with several games running at once. Meant to run for hours (e.g. overnight).

    python games/blue_car/train.py

Speed measured on the desktop (RTX 5070 Ti, 32 GB RAM): 1 game ~6.7 steps/s, 12 games ~62 steps/s.
16 games was faster (~73) but left only ~2.6 GB of RAM free, too little for a long run.
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))   # models/ and runs/ paths below are relative to this game's folder

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecMonitor

from blue_car_env import BlueCarEnv
from core.realtime import InfoStats, RestartAfterUpdate

RUN = "v1"                 # one name per run: used for the model files and the TensorBoard log
GAMES = 12                 # games running at once, each in its own process and browser
STEPS = 1_500_000          # ~55 steps/s with restarts and updates -> about 7.5 hours


def make_env():
    return BlueCarEnv()


if __name__ == "__main__":   # required on Windows: each game process re-imports this file
    # SubprocVecEnv: GAMES games in parallel. VecMonitor: episode reward/length stats (ep_rew_mean).
    # VecFrameStack: the agent sees the last 4 frames, so it can see motion. play.py/test.py must stack 4 too.
    env = VecFrameStack(VecMonitor(SubprocVecEnv([make_env] * GAMES)), n_stack=4)

    callbacks = [
        RestartAfterUpdate(),        # real time: release keys while PPO updates, restart games before each batch
        InfoStats(["cubes"]),        # rollout/cubes = average cubes collected per episode
        # Save a copy every ~50k steps, so a crash in the night doesn't lose everything.
        # There is no EvalCallback: in real time, test episodes would pause the 12 training games
        # (keys still held) for minutes. Test the checkpoints afterwards with test.py instead.
        CheckpointCallback(save_freq=50_000 // GAMES, save_path=f"models/{RUN}_checkpoints/", name_prefix=RUN),
    ]

    # CnnPolicy = a convolutional network for images. n_steps=512 per game -> 6,144 steps per batch (~100 s),
    # then an update of ~9 s on the CPU (a CUDA build of PyTorch would make updates much faster).
    model = PPO("CnnPolicy", env, n_steps=512, verbose=1, tensorboard_log="runs/")

    try:
        model.learn(total_timesteps=STEPS, callback=callbacks, tb_log_name=RUN)
    finally:
        model.save(f"models/{RUN}_final")   # also saves if you stop it with Ctrl+C
        env.close()
