"""Train an agent on Blue Car, with several games running at once. Meant to run for hours (e.g. overnight).

    python games/blue_car/train.py

Speed measured on the desktop (RTX 5070 Ti, 32 GB RAM): 1 game ~6.7 steps/s, 12 games ~62 steps/s.
16 games was faster (~73) but left only ~2.6 GB of RAM free, too little for a long run.
"""
import glob
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))   # models/ and runs/ paths below are relative to this game's folder

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecMonitor

from blue_car_env import VIRTUAL_TIME, BlueCarEnv
from core.demo_ppo import DemoPPO
from core.realtime import InfoStats, RestartAfterUpdate
from train_bc import demo_dataset

RUN = "v10"                # one name per run: used for the model files and the TensorBoard log
GAMES = 10                 # games running at once, each in its own process and browser
STEPS = 1_200_000          # steps to train in THIS run (added on top when resuming). ~40-45 steps/s -> ~7.5-8 h

# Continue training a saved model instead of starting from random weights:
#   None      = start from scratch
#   "latest"  = the newest checkpoint in models/<RUN>_checkpoints/ (e.g. after stopping a run to change a setting)
#   "latest:v3" = the newest checkpoint of another run (continue its model under this run's name)
#   a path    = that model, e.g. "models/v3_checkpoints/v3_100000_steps"
# The step count continues from the checkpoint. Only resume with the same observations/actions:
# a different reward or training setting is fine, a different _get_obs is not.
RESUME = "models/v7e_checkpoints/v7e_3460440_steps"   # v10: v7e near its best (the checkpoint that won 1 of 5
                           # test runs). New in v10: 6 of 7 actions include gas (see ACTIONS), v9's reward.

# v3 settled on "get cube 1, then dawdle safely until the time limit" and stopped exploring (entropy -0.34).
# ENT_COEF: bonus for keeping the action choice random, so it keeps trying new things (SB3 default 0).
# N_STEPS: steps per game per batch. The games restart after every batch, so it must be at least as long as
# an episode (MAX_STEPS in blue_car_env.py, now 1000 so a slow driver still has time to reach cube 2).
ENT_COEF = 0.005          # v3b (0.01) explored mostly by leaving the road (98% off road): halved for v4
LEARNING_RATE = 1e-4      # back to 1e-4: the new actions change a lot, so it has to re-learn steering
N_STEPS = 1024

# Demonstration loss (core/demo_ppo.py): after each PPO update, extra small updates toward Ryan's recorded actions
# (demos/, prepared like train_bc.py), so the copied knowledge of later track sections doesn't fade while it trains
# on the early part. The weight shrinks to 0 over DEMO_DECAY steps: after that it's pure PPO again.
DEMO_WEIGHT = 0.0          # 0 = off. v7c used 0.1: the demo loss dropped to ~0.03 within a few updates (it memorised
                           # the 2,500 recorded steps) and cubes fell 2.82 -> ~0.8 (back to the plain copy's level)
DEMO_DECAY = 500_000


REPLAY_EVERY = 5          # game 1 saves every 5th episode with pictures for viewer.html (the others: nothing)


def make_env(game=0):
    # Only the first game records (runs/<RUN>_replay.jsonl): enough to watch progress, and training stays fast.
    if game == 0:
        return BlueCarEnv(record_file=f"runs/{RUN}_replay.jsonl", record_every=REPLAY_EVERY, games=GAMES)
    return BlueCarEnv()


def newest_checkpoint(run):
    files = glob.glob(f"models/{run}_checkpoints/{run}_*_steps.zip")
    if not files:
        return None
    newest = max(files, key=lambda f: int(f.rsplit("_", 2)[-2]))   # the step number in the file name
    return newest[:-len(".zip")]


if __name__ == "__main__":   # required on Windows: each game process re-imports this file
    # SubprocVecEnv: GAMES games in parallel. VecMonitor: episode reward/length stats (ep_rew_mean).
    # VecFrameStack: the agent sees the last 4 frames, so it can see motion. play.py/test.py must stack 4 too.
    env = VecFrameStack(VecMonitor(SubprocVecEnv([lambda g=g: make_env(g) for g in range(GAMES)])), n_stack=4)

    callbacks = [
        # real time only: release keys while PPO updates, restart games before each batch
        # (with virtual time the games are frozen during updates, so there is nothing to do)
        *([] if VIRTUAL_TIME else [RestartAfterUpdate()]),
        # rollout/cubes = average cubes per episode; lost / off_road = fraction of episodes ending that way
        InfoStats(["cubes", "lost", "off_road", "won", "stalled"]),
        # Save a copy every ~50k steps, so a crash in the night doesn't lose everything.
        # There is no EvalCallback: in real time, test episodes would pause the 12 training games
        # (keys still held) for minutes. Test the checkpoints afterwards with test.py instead.
        CheckpointCallback(save_freq=50_000 // GAMES, save_path=f"models/{RUN}_checkpoints/", name_prefix=RUN),
    ]

    # CnnPolicy = a convolutional network for images. N_STEPS per game -> N_STEPS x GAMES steps per batch
    # (~3 min with 1024), then an update on the GPU (a few seconds with the CUDA build of PyTorch).
    if RESUME and RESUME.startswith("latest"):
        start = newest_checkpoint(RESUME.split(":")[1] if ":" in RESUME else RUN)
    else:
        start = RESUME
    if start:
        print(f"Resuming from {start}")
        # Settings passed here replace the ones saved in the checkpoint.
        model = DemoPPO.load(start, env=env, tensorboard_log="runs/", n_steps=N_STEPS, ent_coef=ENT_COEF, verbose=1,
                             learning_rate=LEARNING_RATE)
    else:
        print("Starting from scratch")
        model = DemoPPO("CnnPolicy", env, n_steps=N_STEPS, ent_coef=ENT_COEF, learning_rate=LEARNING_RATE,
                        verbose=1, tensorboard_log="runs/")
    print(f"n_steps {model.n_steps}, ent_coef {model.ent_coef}, learning_rate {model.learning_rate}, "
          f"virtual time {VIRTUAL_TIME}")
    if DEMO_WEIGHT:
        X, y = demo_dataset()
        model.set_demos(X, y, weight=DEMO_WEIGHT, decay_steps=DEMO_DECAY)
        print(f"demonstration loss: {len(y)} recorded steps, weight {DEMO_WEIGHT} -> 0 over {DEMO_DECAY:,} steps")

    try:
        # reset_num_timesteps=False when resuming: keep counting from the checkpoint's step number
        model.learn(total_timesteps=STEPS, callback=callbacks, tb_log_name=RUN, reset_num_timesteps=not start)
    finally:
        model.save(f"models/{RUN}_final")   # also saves if you stop it with Ctrl+C
        env.close()
