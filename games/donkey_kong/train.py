"""Train an agent on Donkey Kong, with several games running at once. Meant to run for hours (e.g. overnight).

    python games/donkey_kong/train.py

v1, written by Claude on 2026-10-07 from PLAN.md (Ryan may rewrite it). Modelled on games/blue_car/train.py.
Real time: ~7 steps/s per game, so 10 games ~50-60 steps/s (1M steps ~ 5 h). Ctrl+C stops it and still saves.
"""
import glob
import os
import sys
import time
os.chdir(os.path.dirname(os.path.abspath(__file__)))   # models/ and runs/ paths below are relative to this game's folder

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecMonitor

from core.realtime import InfoStats
from donkey_kong_env import DonkeyKongEnv

RUN = "v2c"                # one name per run: used for the model files, the replay and the TensorBoard log
GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 10   # games at once (each its own process + browser), e.g.
                           # `python train.py 2` while playing something heavy (~4 GB instead of ~14 GB), 10 otherwise
STEPS = 3_000_000          # steps to train in THIS run (more than one night: stop it whenever you like)

# Continue training a saved model instead of starting from random weights:
#   None      = start from scratch
#   "latest"  = the newest checkpoint in models/<RUN>_checkpoints/ (e.g. after stopping a run to change a setting)
#   a path    = that model, e.g. "models/v1_checkpoints/v1_100000_steps"
# Only resume with the same observations/actions: a different reward or training setting is fine.
#   "latest:v1" = the newest checkpoint of another run (continue its model under this run's name)
RESUME = "latest"          # v2c continues itself: the newest of its checkpoints and models/v2c_final (a clean stop).
                           # v2c was started from "latest:v2b" (v2b + the step penalty). Earlier: "models/v2_final" (v2b)

ENT_COEF = 0.01            # bonus for keeping the action choice random, so it keeps exploring (SB3 default 0).
                           # Finding the ladders is the hard part here, so a bit more than Blue Car's 0.005.
LEARNING_RATE = 2e-4       # v1b: was 3e-4 (SB3 default): approx_kl grew from 0.01 to 0.21 (healthy: 0.01-0.02),
                           # i.e. every update changed the policy far too much
TARGET_KL = 0.02           # v1b: stop an update early once it has changed the policy this much
                           # v2: learning rate 2e-4 (was 1e-4 for v1b/v1c, which resumed a trained model): a fresh
                           # network needs to move faster; TARGET_KL still caps every update
N_STEPS = 5120 // GAMES    # steps per game per batch: every batch is 5,120 steps whatever GAMES is (10 games: 512
                           # each, ~1.5 min; 2 games: 2,560 each, ~6.5 min), so the updates stay the same size

START_GAP = 3              # seconds between starting the games
REPLAY_EVERY = 50          # game 1 saves every 50th episode with pictures (in runs/<RUN>_episodes.jsonl, with the
                           # episodes worth keeping from all games, see make_env). ~2 MB per saved episode.


class ReleaseKeysDuringUpdate(BaseCallback):
    """PPO stops collecting for a few seconds to update the network, but the games keep running in real time with
    the last keys still held. Let go of all keys meanwhile, so Mario doesn't walk into barrels on autopilot.
    (Unlike Blue Car's RestartAfterUpdate this doesn't restart the games: Donkey Kong has no restart key.)"""

    def _on_rollout_end(self):
        self.training_env.env_method("release_keys")

    def _on_step(self):
        return True


def make_env(game=0):
    time.sleep(game * START_GAP)   # start the games a few seconds apart: 10 page loads at once timed out once
    # All games write ONE file, runs/<RUN>_episodes.jsonl (open it in viewer.html): every REPLAY_EVERY-th episode of
    # game 1 with pictures plus a result line for all its episodes (the chart), and from every game the episodes worth
    # keeping, with "why": first time at a new line, first win, best so far (shared state: runs/<RUN>_best.json).
    return DonkeyKongEnv(record_file=f"runs/{RUN}_episodes.jsonl", record_every=REPLAY_EVERY, games=GAMES,
                         game=game + 1, log_all=(game == 0), shared=True, best_file=f"runs/{RUN}_best.json")


class StopFile(BaseCallback):
    """Stop cleanly when the file runs/<RUN>.stop exists: the current batch finishes, the model is saved
    (models/<RUN>_final, see the end of this file) and the file is removed. Use it to restart with a different
    number of games without losing anything (PowerShell):  New-Item games/donkey_kong/runs/<RUN>.stop
    then start again, e.g.  python games/donkey_kong/train.py 2   (RESUME = "latest" continues from the saved model)"""

    def _on_step(self):
        path = f"runs/{RUN}.stop"
        if os.path.exists(path):
            os.remove(path)
            print(f"{path} found: stopping and saving", flush=True)
            return False
        return True


class TrainStepToGames(BaseCallback):
    """Tell every game the real training step before each batch, so saved episodes show the true step."""

    def _on_rollout_start(self):
        self.training_env.env_method("set_train_step", self.num_timesteps)

    def _on_step(self):
        return True


def newest_checkpoint(run):
    """The newest saved model of a run: its checkpoints and models/<run>_final.zip (saved by a clean stop), by file time."""
    files = glob.glob(f"models/{run}_checkpoints/{run}_*_steps.zip") + glob.glob(f"models/{run}_final.zip")
    if not files:
        return None
    return max(files, key=os.path.getmtime)[:-len(".zip")]


if __name__ == "__main__":   # required on Windows: each game process re-imports this file
    # SubprocVecEnv: GAMES games in parallel. VecMonitor: episode reward/length stats (ep_rew_mean).
    # VecFrameStack: the agent sees the last 4 frames, so it can see which way barrels move. play.py/test.py stack 4 too.
    env = VecFrameStack(VecMonitor(SubprocVecEnv([lambda g=g: make_env(g) for g in range(GAMES)])), n_stack=4)

    callbacks = [
        StopFile(),
        ReleaseKeysDuringUpdate(),
        TrainStepToGames(),
        # rollout/height = average best checkpoint line per episode (the number that matters),
        # died / won = fraction of episodes ending that way
        InfoStats(["height", "cells", "ladders", "died", "won"]),   # cells (v1c), ladders (v2) per life
        # Save a copy every ~50k steps, so a crash in the night doesn't lose everything.
        CheckpointCallback(save_freq=50_000 // GAMES, save_path=f"models/{RUN}_checkpoints/", name_prefix=RUN),
    ]

    if RESUME and RESUME.startswith("latest"):
        start = newest_checkpoint(RESUME.split(":")[1] if ":" in RESUME else RUN)
    else:
        start = RESUME
    if start:
        print(f"Resuming from {start}")
        model = PPO.load(start, env=env, tensorboard_log="runs/", n_steps=N_STEPS, ent_coef=ENT_COEF,
                         learning_rate=LEARNING_RATE, target_kl=TARGET_KL, verbose=1)
    else:
        print("Starting from scratch")
        model = PPO("CnnPolicy", env, n_steps=N_STEPS, ent_coef=ENT_COEF, learning_rate=LEARNING_RATE,
                    target_kl=TARGET_KL, verbose=1, tensorboard_log="runs/")

    try:
        # reset_num_timesteps=False when resuming: keep counting from the checkpoint's step number
        model.learn(total_timesteps=STEPS, callback=callbacks, tb_log_name=RUN, reset_num_timesteps=not start)
    finally:
        model.save(f"models/{RUN}_final")   # also saves if you stop it with Ctrl+C
        env.close()
