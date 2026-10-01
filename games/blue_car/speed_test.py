"""Measure training speed: N games at once (like train.py), random actions, real time vs virtual time.

    python games/blue_car/speed_test.py

Prints total steps per second and free RAM for each setting. Run it when no training is running.
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import subprocess
import sys
import time

import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv

import blue_car_env as m

SECONDS = 60
SETTINGS = [(True, 10), (True, 13), (False, 10)]     # (virtual time?, games)


def free_gb():
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"], capture_output=True, text=True)
    return int(out.stdout.strip()) / 1024 / 1024


def make(vt):
    return lambda: m.BlueCarEnv(virtual_time=vt)


if __name__ == "__main__":
    settings = SETTINGS if len(sys.argv) < 2 else [(a == "vt", int(n)) for a, n in (x.split(":") for x in sys.argv[1:])]
    for vt, n in settings:
        env = SubprocVecEnv([make(vt) for _ in range(n)])
        env.reset()
        steps, t = 0, time.time()
        while time.time() - t < SECONDS:
            env.step(np.random.choice([0, 1, 1, 1, 2, 3], n))     # mostly gas, like a driving agent
            steps += n
        rate = steps / (time.time() - t)
        game_ms = m.STEP_GAME_MS if vt else None
        speed = f" = {rate * m.STEP_GAME_MS / 1000:.1f} game-seconds per second" if vt else ""
        print(f"{'virtual' if vt else 'real   '} time, {n:2} games: {rate:6.1f} steps/s{speed} | free RAM {free_gb():.1f} GB",
              flush=True)
        env.close()
