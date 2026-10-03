"""Gymnasium environment for Blue Car (Unity WebGL, runs in real time, pixels only).

The plumbing is done (open the game, focus, hold keys, wait, screenshot, respawn).
The TODOs are the RL decisions and yours to write: actions, observations, reward, game over.
Find out what you need first with explore.py (see NOTES.md).
"""
import time

import gymnasium as gym
import numpy as np

from core.browser import Browser
from core.replay import ReplayRecorder

import cv2

# ---- settings -----------------------------------------------------------
GAME_URL = "https://html-classic.itch.zone/html/13523181/index.html"
LOAD_SECONDS = 20        # Unity needs a while to load before it reacts to keys
STEP_SECONDS = 1 / 15    # how long each action is held; with the screenshot, one step takes ~0.15 s (~6.7 steps/s)

# Virtual time = frame control (core/browser.py): instead of playing in real time, the game is frozen and runs exactly
# STEP_GAME_MS of game time per step (60 frames per game-second, clock advanced per frame), as fast as the computer
# can: ~3.4x real time with 1 game. The game also stays frozen while PPO updates, so the batch restarts
# (core/realtime.py) are not needed. False = real time (all runs up to v7b). (A first version advanced the clock in
# one jump per step: the game then drew 1 frame per step and steering barely worked; v7b drove 0.8 cubes vs 3.0.)
VIRTUAL_TIME = True
STEP_GAME_MS = 200       # matches the real-time models: v7b_final drove 3.2 cubes with 200 ms steps (6 episodes),
                         # 3.0 in real time; 250 ms gave 1.7
RESPAWN_KEY = "Backspace"
RESPAWN_SECONDS = 0.2    # wait after respawning. Measured: the respawn is done in < 0.1 s (12/12 resets back on
                         # the road at the start after 0.2 s). Keep it short: the games step in lockstep, so
                         # every reset makes all games wait (1.0 s cut training to ~18 steps/s with short episodes).
MAX_STEPS = 2000         # time limit per episode (truncated, not a real ending): 400 s of game time at 200 ms steps.
                         # Was 1000: in v7d, 34 of its 48 best episodes (8+ cubes) hit the limit near the end of the track
                         # (Ryan finishes in ~25 s with 10-13 cubes), so it never reached the finish and its win bonus.
                         # With virtual time the games are frozen during PPO updates, so episodes can span batches.

# Reward (v9): see NOTES.md for why
CUBE_REWARD = 1.0        # the cube counter bottom-left went up
LOSE_PENALTY = 20.0      # the "YOU LOSE" screen appeared, or off the road too long (was 10 until v3)

# v3 learned "get cube 1, then dawdle safely": after cube 1 nothing rewards driving on, and cube 2 is too
# far to find by chance (1 of 92 batches in v3b). Measuring forward movement from the picture (optical flow)
# didn't work (grain + look-alike dashes), so v4 rewards the action instead: gas while on the road.
# ~100 steps of look-ahead -> worth up to ~+5, far more than the one cube it can reach now.
# Can't be farmed by pushing into a wall: the still penalty (-0.1) is bigger than this.
GAS_REWARD = 0.05

# v6: reward driving toward a visible cube. The cubes are the only bright orange/yellow things in the game
# (lots of red and green, little blue): measured on ~70 screenshots, far cube 160-260 pixels, near 500-1,800,
# no cube 0, and nothing else (car, counter, YOU LOSE text, road lines, trees) ever counted.
# Potential-based, like the toy car's v3 progress: phi = sqrt(cube pixels) / 40 (~0.3 far, ~1 just before
# collecting), reward += CUBE_APPROACH * (phi now - phi before). Getting closer pays, turning away costs,
# hovering or wiggling adds up to nothing. When a cube is collected it flares up and vanishes over ~2 steps
# (measured: phi 1.20 -> 0.38 -> 1.18 -> 0): the approach reward is skipped for PICKUP_STEPS steps after a
# pickup, so collecting (+1) stays the real prize.
CUBE_APPROACH = 1.0
PICKUP_STEPS = 6         # ~1 s. Was 3: after a pickup, cube-coloured fragments fly around for a moment (seen in
                         # Ryan's recordings) and the counter digit animates, so one pickup could count twice.
                         # Not much longer: Ryan collects a cube every ~2 s (10-13 per run), the next must still count.

# The finish: "YOU WIN / PRESS BACKSPACE TO RESPAWN" on a bright green screen, found in Ryan's recordings
# (5 of 6 runs reached it, ~20-25 s after the start). Until then the game-over check counted it as losing!
# Winning ends the episode with WIN_REWARD, plus FAST_BONUS for every step under FAST_STEPS (Ryan: 125-155 steps),
# so a faster finish is worth more.
WIN_REWARD = 30.0

# v8: no new cube for CUBE_TIMEOUT steps = stalled = lost (LOSE_PENALTY). v7 (and v1) learned that standing still
# is safest: without a still penalty it only costs -0.01 per step, crashing costs -20. The cube counter is the one
# detector that's reliable everywhere, so "not progressing" is judged by it. Ryan's recordings: gaps between cubes
# median 18 steps, 95% within 36, max 57 (also from the last cube to the finish), so 90 (~15 s) leaves room for
# slower driving but makes standing still exactly as bad as crashing.
# OFF since v8 failed: the agent believed driving leads to a crash (-20 at ~step 40), and stalling only gives -20
# at step 90, which discounting makes ~40% cheaper. So it coasted until the timeout (0-1% gas, cubes 0.1-0.25).
# None = off; a number = the rule is on.
CUBE_TIMEOUT = None
FAST_BONUS = 0.02        # v9: was 0.1 per step under 300, but the agent finished in ~1,300 steps, so it never got any
FAST_STEPS = 1500        # (Ryan ~150 steps -> +27; a 1,300-step finish -> +4): getting faster always pays a bit
# v9: was 0.01. With GAS_REWARD 0.05 every step of driving on the road netted +0.04, so a longer episode paid MORE
# (a slow 1,300-step lap earned ~+52 from gas alone, more than the +30 win): v7e won a few times, then crawled.
# Now driving with gas on the road nets -0.01 per step (time costs), standing / coasting -0.06 (much worse).
STEP_PENALTY = 0.06

# v1 learned to stand still: not moving never loses, so it beat driving and losing (-10).
# v2 punishes every step where the picture barely changed.
# Size: with gamma = 0.99 the agent looks ~100 steps ahead, so 100 still steps cost ~-10 (about -6 discounted),
# about as bad as losing. Much bigger (e.g. 1 per step) makes crashing on purpose (-10 once, then the
# episode is over) the cheapest way out whenever the car stops, so it would learn to crash instead.
# v5: turned OFF (0). The "still" check misfires while driving: in darker parts of the track normal driving
# only changes the picture by ~23 per pixel (median), and 39% of gas-on-road steps counted as "still".
# So it punished normal driving (-0.1 vs +0.05 gas) and pushed v4 to swerve (big picture change) off the road.
# Standing still is now covered by the gas reward (standing earns nothing, driving on the road +0.04/step)
# and hiding by the off-road rule. Watch for: pushing into a wall with the road still in view to farm gas.
STANDING_STILL_PENALTY = 0.0

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
# Recalibrated on Ryan's recordings of the whole track (283 on-road frames): the old rule (light gray lines)
# found no lines at all in several later sections, where the lines are darker and bluer, so it would have
# ended every episode there. Now: bluish line pixels (see _on_road). On the road: median ~5,200, 5th
# percentile 1,340, and never below 800 for 10 steps in a row; off the road (20 frames): at most 467.
ROAD_THRESHOLD = 800
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
# The best models (v7e) use these actions. v10 tried 6 of 7 actions with gas (left/right/coast2 -> gas versions,
# since the agent pressed gas on only 21-31% of steps): it threw away too much of v7e's driving (8 -> ~0.7 cubes in
# 2.7 h) and was stopped. Its models need that action set.


class BlueCarEnv(gym.Env):
    def __init__(self, headless=True, record_file=None, record_every=5, games=1, virtual_time=None):
        """record_file: save episodes for viewer.html (runs/<run>_replay.jsonl); frames for every record_every-th
        episode, a result line for all. games: how many games train at once (to estimate the training step)."""
        super().__init__()
        self.recorder = ReplayRecorder(record_file, record_every, games, meta={"game": "blue_car",
                                       "actions": ["coast", "gas", "gas+left", "gas+right", "left", "right", "coast"]}
                                       ) if record_file else None
        # virtual_time: None = the VIRTUAL_TIME setting; record_demo.py / play.py pass False (real time)
        self.virtual_time = (VIRTUAL_TIME if virtual_time is None else virtual_time) and headless
        self.browser = Browser(GAME_URL, headless=headless, viewport=(960, 600), frame_control=self.virtual_time)
        self.browser.wait(LOAD_SECONDS)
        self.browser.page.mouse.click(480, 280)       # the game only gets keys after a click (focus): canvas centre
        if self.virtual_time:
            self.browser.wait(0.5)
            self.browser.freeze()

        self.action_space = gym.spaces.Discrete(len(ACTIONS))

        # v6: 2 channels: the gray road picture + a cube mask (see _get_obs)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 2), dtype=np.uint8)

        self.prev_frame = None
        self.steps = 0
        self.cubes = 0
        self.off_road_steps = 0
        self.pickup_steps_left = 0
        self.collected = False
        self.total_reward = 0.0
        self.cubes_before_reset, self.reward_before_reset = 0, 0.0
        self.steps_since_cube = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.browser.hold([])
        self.browser.press(RESPAWN_KEY)   # restarts at any time, lost or not (checked with explore.py)
        self._wait(RESPAWN_SECONDS)

        frame = self.browser.screenshot()
        self.prev_frame = frame
        self.steps = 0
        self.cubes = 0
        self.off_road_steps = 0
        self.pickup_steps_left = 0
        self.total_reward = 0.0
        self.steps_since_cube = 0
        if self.recorder:
            if self.recorder.active and self.recorder.episode_steps:   # reset mid-episode: cut by a batch restart
                self.recorder.end_episode({"ended": "restart", "cubes": self.cubes_before_reset,
                                           "reward": round(self.reward_before_reset, 2)})
            self.recorder.start_episode()
        return self._get_obs(frame), {}

    def step(self, action):
        self.browser.hold(ACTIONS[action])
        if self.virtual_time:
            frame = self.browser.advance(STEP_GAME_MS, screenshot=True)   # exactly this much game time, then frozen
        else:
            time.sleep(STEP_SECONDS)          # real time: the game runs while the keys are held
            frame = self.browser.screenshot()
        self.steps += 1

        # Off the road: count steps in a row without road markings, reset the count when back on the road.
        on_road = self._on_road(frame)
        self.off_road_steps = 0 if on_road else self.off_road_steps + 1
        off_road = self.off_road_steps >= OFF_ROAD_STEPS
        won = self._is_win(frame)
        game_over = self._is_game_over(frame)               # YOU LOSE only (not the green YOU WIN screen)
        if won:
            off_road = False                                # the green win screen has no road lines

        obs = self._get_obs(frame)
        reward = self._get_reward(self.prev_frame, frame)   # includes -LOSE_PENALTY if the game says YOU LOSE
        if off_road and not game_over:
            reward -= LOSE_PENALTY                          # off the road too long: counts as losing
        if on_road and "w" in ACTIONS[action]:
            reward += GAS_REWARD                            # v4: driving forward on the road
        if won:
            reward += WIN_REWARD + FAST_BONUS * max(0, FAST_STEPS - self.steps)
        self.steps_since_cube = 0 if self.collected else self.steps_since_cube + 1   # collected: set by _get_reward
        stalled = (CUBE_TIMEOUT is not None and not (won or game_over or off_road)
                   and self.steps_since_cube >= CUBE_TIMEOUT)
        if stalled:
            reward -= LOSE_PENALTY                          # v8: no progress for too long counts as losing
        terminated = game_over or off_road or won or stalled
        truncated = not terminated and self.steps >= MAX_STEPS
        self.cubes += self.collected
        info = {"cubes": self.cubes, "lost": game_over, "off_road": off_road, "won": won, "stalled": stalled}

        self.total_reward += reward
        self.cubes_before_reset, self.reward_before_reset = self.cubes, self.total_reward
        if self.recorder:
            self.recorder.add_step(frame[:572, :945], {"a": int(action), "r": round(float(reward), 3),
                                   "c": self.cubes, "road": bool(on_road)})
            if terminated or truncated:
                ended = ("won" if won else "lost" if game_over else "off road" if off_road
                         else "stalled" if stalled else "time limit")
                self.recorder.end_episode({"ended": ended, "cubes": self.cubes, "reward": round(self.total_reward, 2),
                                           "length": self.steps})

        self.prev_frame = frame
        return obs, reward, terminated, truncated, info

    def _wait(self, seconds):
        """Let the game run for this long: real time, or that much virtual game time."""
        if self.virtual_time:
            self.browser.advance(int(seconds * 1000))
        else:
            time.sleep(seconds)

    def release_keys(self):
        """Let go of all keys (used by core.realtime.RestartAfterUpdate while PPO updates)."""
        self.browser.hold([])

    def close(self):
        if self.recorder:
            self.recorder.close()
        self.browser.close()


    def _get_obs(self, frame):
        img = frame[:572, :945] #Creates Rows and Columns
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) #Turns it to grey scale
        img = cv2.equalizeHist(img) #spreads the dark values over the full 0–255 range increases the contrast

        img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA) #Resize the image to 84 by 84

        # v6: second channel = where the cubes are. In the gray picture a far cube is only 1-2 grainy pixels,
        # so mark every 84x84 pixel that contains any cube pixel as fully white (255), everything else black.
        cubes = self._cube_mask(frame).astype(np.uint8) * 255
        cubes = cv2.resize(cubes, (84, 84), interpolation=cv2.INTER_AREA)
        cubes = np.where(cubes > 0, 255, 0).astype(np.uint8)

        return np.stack([img, cubes], axis=2) #(84, 84, 2): channel 0 = road picture, channel 1 = cubes

        

    def _get_reward(self, prev_frame, frame):
        reward = -STEP_PENALTY

        if STANDING_STILL_PENALTY:   # off since v5 (it misfired while driving); only computed if turned back on
            old = self._get_obs(prev_frame)[:, :, 0].astype(int) #int, because uint8 wraps around: 3 - 5 = 254
            new = self._get_obs(frame)[:, :, 0].astype(int)
            difference = np.abs(new - old).mean() #average change per pixel, 0 = identical
            if difference < STILL_THRESHOLD:
                reward -= STANDING_STILL_PENALTY

        self.collected = False
        if self.pickup_steps_left > 0:   # just picked one up: the cube is still flaring up / breaking apart
            self.pickup_steps_left -= 1
        elif self._cube_collected(prev_frame, frame):
            reward += CUBE_REWARD
            self.collected = True
            self.pickup_steps_left = PICKUP_STEPS
        else:   # v6: closer to a visible cube
            reward += CUBE_APPROACH * (self._cube_potential(frame) - self._cube_potential(prev_frame))
        if self._is_game_over(frame):
            reward -= LOSE_PENALTY
            
        
        return reward

    def _cube_mask(self, frame):
        """True where a cube is: bright orange/yellow (lots of red and green, little blue). See CUBE_APPROACH."""
        g = frame[:572, :945].astype(int)
        r, gr, b = g[..., 0], g[..., 1], g[..., 2]
        return (r > 150) & (gr > 90) & (b < 90) & (r > b + 100)

    def _cube_potential(self, frame):
        """How close the visible cube(s) are: sqrt(cube pixels) / 40 -> ~0.3 far, ~1 just before collecting, 0 if none."""
        return float(np.sqrt(self._cube_mask(frame).sum()) / 40)

    def _cube_collected(self, prev_frame, frame):
        """True if the cube counter bottom-left changed since the last frame.
        The digit is white (all colours > 200) on a dark background: compare those white pixels.
        Measured: a digit change flips ~1,200 of them, screen grain at most ~10. Respawn resets the
        counter within 0.1 s, before reset()'s first screenshot, so it never counts as a cube."""
        before = (prev_frame[470:572, 10:200] > 200).all(axis=2)
        after = (frame[470:572, 10:200] > 200).all(axis=2)
        return bool((before != after).sum() > 200)

    def _on_road(self, frame):
        """True if road lines are visible in front of the car (see ROAD_THRESHOLD).
        Bluish line pixels in the lower part of the screen, right of the cube counter: fairly blue, a bit bluer
        than red, not black. The bright blue car (blue > 170) doesn't count."""
        low = frame[300:572, 200:945].astype(int)
        r, g, b = low[..., 0], low[..., 1], low[..., 2]
        lines = ((b > 50) & (b > r + 10) & (r > 15) & (b < 170)).sum()
        return bool(lines > ROAD_THRESHOLD)

    def _end_text(self, frame):
        """True if the big white "YOU LOSE / YOU WIN ... PRESS BACKSPACE TO RESPAWN" text is showing."""
        middle = frame[250:350, 100:860]
        white = (middle > 200).all(axis=2)
        return bool(white.sum() > 2000)   # bool(): numpy gives numpy.bool_, Gymnasium needs a real bool

    def _green(self, frame):
        """Share of the screen that is bright green: the YOU WIN screen is 78-95%, normal driving at most 21%."""
        g = frame[:572, :945].astype(int)
        return float((g[..., 1] - g[..., 2] - g[..., 0] // 2 > 60).mean())

    def _is_win(self, frame):
        return self._end_text(frame) and self._green(frame) > 0.5

    def _is_game_over(self, frame):
        return self._end_text(frame) and self._green(frame) <= 0.5

