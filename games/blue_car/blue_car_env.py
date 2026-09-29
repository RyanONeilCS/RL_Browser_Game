"""Gymnasium environment for Blue Car (Unity WebGL, runs in real time, pixels only).

The plumbing is done (open the game, focus, hold keys, wait, screenshot, respawn).
The TODOs are the RL decisions and yours to write: actions, observations, reward, game over.
Find out what you need first with explore.py (see NOTES.md).
"""
import time

import gymnasium as gym
import numpy as np

from core.browser import Browser

import cv2

# ---- settings -----------------------------------------------------------
GAME_URL = "https://html-classic.itch.zone/html/13523181/index.html"
LOAD_SECONDS = 20        # Unity needs a while to load before it reacts to keys
STEP_SECONDS = 1 / 15    # how long each action is held; with the screenshot, one step takes ~0.15 s (~6.7 steps/s)
RESPAWN_KEY = "Backspace"
RESPAWN_SECONDS = 1.0    # wait after respawning before the next step (TODO: check how long it really needs)
MAX_STEPS = 500          # time limit per episode (truncated, not a real ending): ~100 s with 12 games at ~5 steps/s,
                         # about one PPO batch (n_steps=512 in train.py), since games restart after every batch

# Reward (v1): see NOTES.md for why
CUBE_REWARD = 1.0        # the cube counter bottom-left went up
LOSE_PENALTY = 10.0      # the "YOU LOSE" screen appeared
STEP_PENALTY = 0.01      # per step, so standing still isn't free (discounted, at most -1 in total: < LOSE_PENALTY)

# Action number -> keys held during that step.
# Found with explore.py (see NOTES.md): w = gas, steering only works while moving,
# s = reverse: loses right at the start, but is fine later (e.g. to back out of trees).
ACTIONS = [
    [],                  # 0 coast
    ["w"],               # 1 gas
    ["w", "a"],          # 2 gas + left
    ["w", "d"],          # 3 gas + right
    ["a"],               # 4 left (while rolling)
    ["d"],               # 5 right (while rolling)
    ["s"],               # 6 brake / reverse
]


class BlueCarEnv(gym.Env):
    def __init__(self, headless=True):
        super().__init__()
        self.browser = Browser(GAME_URL, headless=headless, viewport=(960, 600))
        time.sleep(LOAD_SECONDS)
        self.browser.page.locator("canvas").click()   # the game only gets keys after a click (focus)

        self.action_space = gym.spaces.Discrete(len(ACTIONS))

        # TODO: the shape of what _get_obs returns, e.g. an 84x84 grayscale image:
        #       gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

        self.prev_frame = None
        self.steps = 0
        self.cubes = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.browser.hold([])
        self.browser.press(RESPAWN_KEY)   # restarts at any time, lost or not (checked with explore.py)
        time.sleep(RESPAWN_SECONDS)

        frame = self.browser.screenshot()
        self.prev_frame = frame
        self.steps = 0
        self.cubes = 0
        return self._get_obs(frame), {}

    def step(self, action):
        self.browser.hold(ACTIONS[action])
        time.sleep(STEP_SECONDS)          # real time: the game runs while the keys are held
        frame = self.browser.screenshot()
        self.steps += 1

        obs = self._get_obs(frame)
        reward = self._get_reward(self.prev_frame, frame)
        terminated = self._is_game_over(frame)
        truncated = not terminated and self.steps >= MAX_STEPS
        self.cubes += self._cube_collected(self.prev_frame, frame)
        info = {"cubes": self.cubes, "lost": terminated}

        self.prev_frame = frame
        return obs, reward, terminated, truncated, info

    def release_keys(self):
        """Let go of all keys (used by core.realtime.RestartAfterUpdate while PPO updates)."""
        self.browser.hold([])

    def close(self):
        self.browser.close()


    def _get_obs(self, frame):
        img = frame[:572, :945] #Creates Rows and Columns
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) #Turns it to grey scale
        img = cv2.equalizeHist(img) #spreads the dark values over the full 0–255 range increases the contrast

        img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA) #Crop the image into 84 by 84
        img = img[:, :, None] #(84, 84) -> (84, 84, 1)
        
        return img

        

    def _get_reward(self, prev_frame, frame):
        reward = -STEP_PENALTY
        if self._cube_collected(prev_frame, frame):
            reward += CUBE_REWARD
        if self._is_game_over(frame):
            reward -= LOSE_PENALTY
        return reward

    def _cube_collected(self, prev_frame, frame):
        """True if the cube counter bottom-left changed since the last frame.
        The digit is white (all colours > 200) on a dark background: compare those white pixels.
        Measured: a digit change flips ~1,200 of them, screen grain at most ~10. Respawn resets the
        counter within 0.1 s, before reset()'s first screenshot, so it never counts as a cube."""
        before = (prev_frame[470:572, 10:200] > 200).all(axis=2)
        after = (frame[470:572, 10:200] > 200).all(axis=2)
        return bool((before != after).sum() > 200)

    def _is_game_over(self, frame):
        middle = frame[250:350, 100:860]
        white = (middle > 200).all(axis=2)
        
        return bool(white.sum() > 2000)   # bool(): numpy gives numpy.bool_, Gymnasium needs a real bool

