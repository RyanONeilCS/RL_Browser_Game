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
RESPAWN_SECONDS = 0.2    # wait after respawning. Measured: the respawn is done in < 0.1 s (12/12 resets back on
                         # the road at the start after 0.2 s). Keep it short: the games step in lockstep, so
                         # every reset makes all games wait (1.0 s cut training to ~18 steps/s with short episodes).
MAX_STEPS = 1000         # time limit per episode (truncated, not a real ending): ~3 min at ~5.5 steps/s.
                         # Was 500: v3 often ran out of time dawdling before cube 2. Must not be longer than
                         # one PPO batch (N_STEPS in train.py), since the games restart after every batch.

# Reward (v4): see NOTES.md for why
CUBE_REWARD = 1.0        # the cube counter bottom-left went up
LOSE_PENALTY = 20.0      # the "YOU LOSE" screen appeared, or off the road too long (was 10 until v3)

# v3 learned "get cube 1, then dawdle safely": after cube 1 nothing rewards driving on, and cube 2 is too
# far to find by chance (1 of 92 batches in v3b). Measuring forward movement from the picture (optical flow)
# didn't work (grain + look-alike dashes), so v4 rewards the action instead: gas while on the road.
# ~100 steps of look-ahead -> worth up to ~+5, far more than the one cube it can reach now.
# Can't be farmed by pushing into a wall: the still penalty (-0.1) is bigger than this.
GAS_REWARD = 0.05
STEP_PENALTY = 0.01      # per step, so standing still isn't free (discounted, at most -1 in total: < LOSE_PENALTY)

# v1 learned to stand still: not moving never loses, so it beat driving and losing (-10).
# v2 punishes every step where the picture barely changed.
# Size: with gamma = 0.99 the agent looks ~100 steps ahead, so 100 still steps cost ~-10 (about -6 discounted),
# about as bad as losing. Much bigger (e.g. 1 per step) makes crashing on purpose (-10 once, then the
# episode is over) the cheapest way out whenever the car stops, so it would learn to crash instead.
STANDING_STILL_PENALTY = 0.1

# "Barely changed" = the average change per pixel between the last two 84x84 observations (0 = identical).
# Never 0 here: the game's film grain (made stronger by equalizeHist) changes every frame.
# Measured: standing still 15-17 (also while steering, since steering does nothing when stopped),
# driving 23-72 (median 37). 20 sits in the gap. If the penalty fires while clearly driving, blur the
# images before comparing (removes grain, widens the gap).
STILL_THRESHOLD = 20

# v2 learned to hide: drive off the road into a dark spot where the game never says YOU LOSE, and wiggle
# (in a near-black picture, equalizeHist blows the grain up so much that it doesn't count as "still").
# v3: being off the road for OFF_ROAD_STEPS steps in a row ends the episode and costs LOSE_PENALTY,
# so hiding is exactly as bad as crashing.
# "On the road" = light road markings (edge lines, centre dashes) visible in the lower part of the screen.
# Measured on 30 frames: on the road 443-2,465 marking pixels, half off 28, off the road 0 (every frame).
# But further along the track (darker sections, or the car turned sideways) the road only gives 110-190,
# so 150 ended episodes as "off road" while the car was still on the asphalt, punishing it for driving on.
# 60: still far above every real off-road frame (0), below the dim road.
ROAD_THRESHOLD = 60
OFF_ROAD_STEPS = 10      # ~1.5-2 s in a row: short slips at the edge (e.g. in a corner) are allowed

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
    [],                  # 6 was ["s"] (brake / reverse): removed in v4, reversing never helped. Kept as a
                         #   duplicate of coast so there are still 7 actions and v3 models can be loaded.
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
        self.off_road_steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.browser.hold([])
        self.browser.press(RESPAWN_KEY)   # restarts at any time, lost or not (checked with explore.py)
        time.sleep(RESPAWN_SECONDS)

        frame = self.browser.screenshot()
        self.prev_frame = frame
        self.steps = 0
        self.cubes = 0
        self.off_road_steps = 0
        return self._get_obs(frame), {}

    def step(self, action):
        self.browser.hold(ACTIONS[action])
        time.sleep(STEP_SECONDS)          # real time: the game runs while the keys are held
        frame = self.browser.screenshot()
        self.steps += 1

        # Off the road: count steps in a row without road markings, reset the count when back on the road.
        on_road = self._on_road(frame)
        self.off_road_steps = 0 if on_road else self.off_road_steps + 1
        off_road = self.off_road_steps >= OFF_ROAD_STEPS
        game_over = self._is_game_over(frame)

        obs = self._get_obs(frame)
        reward = self._get_reward(self.prev_frame, frame)   # includes -LOSE_PENALTY if the game says YOU LOSE
        if off_road and not game_over:
            reward -= LOSE_PENALTY                          # off the road too long: counts as losing
        if on_road and "w" in ACTIONS[action]:
            reward += GAS_REWARD                            # v4: driving forward on the road
        terminated = game_over or off_road
        truncated = not terminated and self.steps >= MAX_STEPS
        self.cubes += self._cube_collected(self.prev_frame, frame)
        info = {"cubes": self.cubes, "lost": game_over, "off_road": off_road}

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
        old = self._get_obs(prev_frame).astype(int) #int, because uint8 wraps around: 3 - 5 = 254
        new = self._get_obs(frame).astype(int)
        difference = np.abs(new - old).mean() #average change per pixel, 0 = identical
        
        if self._cube_collected(prev_frame, frame):
            reward += CUBE_REWARD
        if difference < STILL_THRESHOLD:
            reward -= STANDING_STILL_PENALTY
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

    def _on_road(self, frame):
        """True if road markings are visible in front of the car (see ROAD_THRESHOLD).
        Light gray / white-blue pixels in the lower part of the screen, right of the cube counter.
        The blue car itself has almost no red or green, so it doesn't count."""
        low = frame[300:572, 200:945].astype(int)
        r, g, b = low[..., 0], low[..., 1], low[..., 2]
        markings = ((r > 45) & (g > 45) & (b > 70)).sum()
        return bool(markings > ROAD_THRESHOLD)

    def _is_game_over(self, frame):
        middle = frame[250:350, 100:860]
        white = (middle > 200).all(axis=2)
        
        return bool(white.sum() > 2000)   # bool(): numpy gives numpy.bool_, Gymnasium needs a real bool

