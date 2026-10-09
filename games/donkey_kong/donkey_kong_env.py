"""Gymnasium environment for Donkey Kong on freekong.org (a Flash game played by Ruffle, runs in real time, pixels only).

v1, written by Claude on 2026-10-07 from PLAN.md so a first training could run overnight (Ryan may rewrite it).
Every detector below was measured on real screenshots first (numbers in the comments and in NOTES.md).

Episodes: one episode = one life. Because the 10 training games step together (lockstep), reset() never waits for
the game: when Mario dies the episode ends at once, the next episode starts on whatever is on screen (the "25 m"
screen, the Game Over screen, the intro...) and those waiting steps just give reward 0 until Mario is back in play.
Waiting inside reset() would make all 10 games wait for the slowest one (Blue Car lost half its speed that way).
"""
import json
import time

import cv2
import gymnasium as gym
import numpy as np

from core.browser import Browser
from core.replay import ReplayRecorder

# ---- page / plumbing ------------------------------------------------------
GAME_URL = "https://freekong.org/"
GAME_SELECTOR = "#game-container"   # the element the game is drawn in. Ruffle draws inside its own <ruffle-player>
                                    # element, so the default "canvas" selector finds nothing in the page.
VIEWPORT = (530, 490)    # the Flash game's own size (530x490 at 30 fps): the player fills the window, so this
                         # gives 1 game pixel = 1 screenshot pixel (smaller and faster than a big window)
LOAD_SECONDS = 8         # page + Ruffle + the .swf
KEY_SECONDS = 0.15       # how long tap() holds a key. The game checks keys once per frame (30 fps): a
                         # Playwright press() (down + up within ~1 ms) is usually missed, a 0.15 s hold works.
STEP_SECONDS = 1 / 15    # how long each action is held before the screenshot (~67 ms more), so ~7 steps/s per game

# Ads: the page loads Google ad / tracking scripts (Brave blocks them when you play yourself; Playwright's Chromium
# doesn't). They run in all 10 browsers and are the main suspect for the memory leak. Blocked by sending their hosts
# to 0.0.0.0 at startup (Chromium's --host-resolver-rules). Tested: the game still loads, starts on Enter and takes
# keys; all 5 ad / tracking hosts the page used failed, the game's own hosts and fonts loaded.
# (Playwright's page.route was tried first: any request interception broke the game's keyboard.)
AD_HOSTS = ["googlesyndication.com", "doubleclick.net", "googletagmanager.com", "googletagservices.com",
            "google-analytics.com", "adservice.google.com", "pub.network", "inmobi.com", "plausible.io",
            "fundingchoicesmessages.google.com", "amazon-adsystem.com"]
BLOCK_ADS = ["--host-resolver-rules=" + ", ".join(f"MAP {h} 0.0.0.0, MAP *.{h} 0.0.0.0" for h in AD_HOSTS)]

# Page elements drawn on top of the game: the "Click on the game area and press ENTER to start" bar, the menu
# button (top left) and the full-screen button (top right). They cover the score, so hide them.
OVERLAYS = ["#header-message", "#menuToggle", "#full-screen-button", "nav", "aside"]

# ---- actions (PLAN.md) ------------------------------------------------------
# No "do nothing": Blue Car learned to stall. Waiting is still possible (jump in place, step left/right).
# Never Escape: it quits to the main menu.
ACTIONS = [
    ["ArrowLeft"],               # 0 walk left
    ["ArrowRight"],              # 1 walk right
    ["ArrowUp"],                 # 2 climb up
    ["ArrowDown"],               # 3 climb down
    ["Space"],                   # 4 jump in place
    ["ArrowLeft", "Space"],      # 5 jump left
    ["ArrowRight", "Space"],     # 6 jump right
]
ACTION_NAMES = ["left", "right", "up", "down", "jump", "jump left", "jump right"]

# ---- reward v1 (PLAN.md) -----------------------------------------------------
START_ROW = 458          # Mario's feet on the bottom girder at the start (measured 455-460)
CHECKPOINT_SPACING = 25  # a height line every 25 px (plan: 20-30). Girders are ~60 px apart -> 2-3 lines per gap.
                         # A jump lifts Mario ~27 px, so a jump can grab the next line early: harmless, each line
                         # pays only once per life, so a climb is worth the same in total.
CHECKPOINT_REWARD = 1.0  # per new line, once per life
# v1c: exploration reward. v1b climbed the first ladder, then wandered on the 2nd girder: the next working ladder is a
# long walk away and walking sideways earns no height, so nothing told it which way to go (stuck at line ~3 for ~3 h).
# Now the screen is a grid of cells and the first visit to each cell in a life pays a little. Walking the whole 2nd
# girder (~10 cells) is worth about one height line, so climbing stays the main goal. Can't be farmed: once per cell
# per life. Same on every stage (no ladder positions needed).
EXPLORE_CELL = (40, 30)  # cell width, height in pixels
EXPLORE_REWARD = 0.1     # per new cell, once per life
GROUND_STEPS = 6         # the cell uses Mario's standing height = his lowest feet position over the last 6 steps
                         # (~0.85 s, longer than a ~0.7 s jump), so jumping along a girder doesn't pay double
STEP_PENALTY = 0.01      # v2c: per step while Mario is alive (not on the waiting screens between lives). v2b found a
                         # safe pocket at the end of the 3rd girder (jump the barrels as they roll off, often after the
                         # hammer, during which Mario can't climb) and waited there: the eventual death was 600+ steps
                         # away, worth almost nothing after discounting, while climbing risked -10 soon. Now waiting
                         # 100 steps costs ~1 (one height line); rushing into a barrel still costs much more (-10).
DEATH_PENALTY = 10.0     # well above one checkpoint: rushing up a ladder into a barrel must be a bad bet
WIN_REWARD = 30.0        # reaching Pauline (more than a death)
WIN_TIME_BONUS = 45.0    # + up to this much for finishing fast: WIN_TIME_BONUS * (1 - steps / MAX_STEPS)
                         # v1c: was 10. With the exploration reward, every step before reaching Pauline must cost more
                         # than exploring earns, or it would wander instead of finishing: exploring pays ~0.1 per ~5
                         # steps = ~0.02/step; this costs 45 / 1500 = 0.03/step. Also far more than all the cells
                         # together (~100 reachable cells x 0.1 = ~10).
                         # (the plan's "bonus timer left", counted in steps instead of reading the BONUS digits)
WIN_ROW = 100            # Mario's feet above this row = he reached Pauline's platform. Kong's girder top is at
                         # row ~111. NOT CHECKED YET (nothing has reached the top): see NOTES.md.
BEST_MARGIN = 1.0        # a "best so far" episode must beat the previous best reward by this much (else early in
                         # training nearly every life is a tiny new best, ~2 MB of pictures each)
MIN_RECORD_LINE = 2      # new lines / best episodes below this aren't saved (random play reaches line 1 by jumping)
MAX_STEPS = 1500         # time limit per episode (truncated): ~3.5 min of game time at ~7 steps/s
SPAWN_ROW = 430          # a new life only starts from a Mario seen on the bottom girder (feet below this row)
TRACK_DIST = 40          # after that, only blobs within this many pixels (|dx| + |dy|) of Mario's last position count,
TRACK_GROW = 15          # + this much per step he wasn't found (climbing / next to the fire he can be missed).
                         # Mario moves ~6 px per step walking, a jump ~27 px up. Without tracking, barrels (blue and
                         # orange) passed as Mario and the random agent "climbed" 150 px in 5 steps.
TRACK_MAX = 100          # the search never grows beyond this
MAX_RISE = 32            # v1b: Mario's feet can't be more than this above his last position (a jump is ~27 px) ...
CLIMB_PER_STEP = 6       # ... + this per step he wasn't found (climbing a ladder while not recognised) ...
MAX_RISE_LOST = 70       # ... but never more than this (~1 girder). v1 allowed up to 100 px in any direction, and
                         # the tracker climbed onto a fireball in a few hops while Mario stood on the bottom girder.
REACQUIRE_AFTER = 10     # after this many steps without Mario, accept a Mario blob anywhere at or below his last height
MIN_NAVY = 50            # Mario blobs need at least this many navy and red pixels (see mario_candidates)
MIN_RED = 30

# ---- v2: observation ---------------------------------------------------
# v1-v1c shrank the whole 530x490 screen to 84x84: a ladder was ~2 px wide and broken / working ladders looked alike,
# and v1c stayed stuck at the 2nd girder for ~12 h. v2 crops the play area (girders x 70-475, Pauline's platform from
# y ~67 to the bottom girder y ~470) and shows it at 140x140 (1 px = 3 game px), ~3x the detail.
OBS_CROP = (60, 50, 480, 470)   # x0, y0, x1, y1 in game pixels (420x420; drops the score / lives strip)
OBS_SIZE = 140

# ---- v2: ladder reward (barrel stage only) --------------------------------
# A bonus the first time in a life that Mario stands at the bottom of each WORKING ladder, so it learns where the way up
# is (v1c wandered on the 2nd girder). Found on a clean screenshot: 9 working ladders, the broken stubs are left out
# (x 213 on the bottom girder, 192 between girders 3-4, 379 between 4-5, 235 between 5-6).
# (centre x, feet row at the ladder's foot) -- the foot = the girder the ladder starts from.
LADDERS = [
    (407, 447),              # girder 1 -> 2 (bottom girder, right)
    (134, 387), (249, 395),  # girder 2 -> 3
    (278, 335), (407, 328),  # girder 3 -> 4
    (134, 269), (206, 272),  # girder 4 -> 5
    (407, 209),              # girder 5 -> 6 (Kong's girder)
    (307, 161),              # girder 6 -> Pauline
]
LADDER_GIRDER = [0, 1, 1, 2, 2, 3, 3, 4, 5]   # which girder each ladder in LADDERS starts from (0 = bottom)
GIRDER_ROWS = [456, 395, 338, 276, 218, 161]  # girder tops (feet row) mid-screen, bottom to top (each slopes ~10-15)
LADDER_REWARD = 1.0      # once per ladder per life (= one height line)
# v2b: reward walking TOWARD the right ladder, not only arriving. v2 reached the foot of the first 1-2 ladders but
# plateaued at the 3rd girder (~3.7 for 7+ h): the working ladders there are far apart and nothing guided the walk.
# Each step: +LADDER_APPROACH per APPROACH_SCALE px Mario got closer to the nearest working ladder that starts from the
# girder he stands on (minus as much for moving away), so walking back and forth adds up to nothing. Steps where the
# girder or the target ladder changes pay nothing (no penalty for arriving on a new girder far from its ladder).
LADDER_APPROACH = 1.0    # walking APPROACH_SCALE px toward the ladder = +1 (about one height line) ...
APPROACH_SCALE = 300     # ... (a girder is ~400 px wide)
GIRDER_TOL = 25          # Mario counts as on a girder when his standing feet are within this of its row
LADDER_X_TOL = 10        # Mario's centre within this many px of the ladder's centre ...
LADDER_Y_TOL = 12        # ... and his standing feet within this many px of the ladder's foot
# NOTE: only right for the barrel stage (L=01). The other stages need their own list (and a stage check) before the
# agent gets there.

# ---- screen detectors (measured on 150+ screenshots, see NOTES.md) ----------
BONUS_BOX_WHITE = 400    # white pixels of the BONUS box (top right, rows 50-105, x 380-500): ~790-800 while playing,
                         # 0 during the intro, the "25 m" screen and Game Over. (Counting girders instead also fired
                         # during the intro, where Kong carries Pauline up the ladders and she passed as Mario.)
GAME_OVER_WHITE = 2500   # white pixels in rows 120-220: Game Over ~3,800 ("Game Over" text); otherwise < 720
MENU_LOGO = 3000         # orange logo pixels in rows 10-100: main menu ~11,700; otherwise < 130
RESTART_BROWSER_MINUTES = 30   # each game's browser leaks memory (v1's first night: the 10 browsers grew ~1.6 GB an hour,
                               # available RAM fell ~2.4 GB an hour). So every 30 min, at the next reset after Mario died,
                               # the browser is closed and opened again (~9 s, the other games wait meanwhile: ~5% slower).
RETRY_COOLDOWN = 2.0     # seconds between Enter presses on the Game Over screen / menu (Enter starts a new game)


def prepare_page(browser):
    """After loading: hide the page's own elements that sit on top of the game (see OVERLAYS), then click the
    game once so it gets the keys. The click is near the bottom edge, away from the menu text."""
    browser.js("""sels => { for (const s of sels) for (const e of document.querySelectorAll(s))
                              e.style.setProperty('display', 'none', 'important'); }""", OVERLAYS)
    browser.page.mouse.click(VIEWPORT[0] // 2, VIEWPORT[1] - 10)
    time.sleep(0.3)


def tap(browser, key, seconds=KEY_SECONDS):
    """Press a key long enough for the game to notice it (see KEY_SECONDS)."""
    browser.hold([key])
    time.sleep(seconds)
    browser.hold([])


def is_playing(frame):
    """A stage is being played: the BONUS box is showing (also during Mario's death animation, not in the intro)."""
    return int((frame[50:105, 380:500].min(axis=2) > 180).sum()) > BONUS_BOX_WHITE


def is_game_over(frame):
    return (not is_playing(frame)) and int((frame[120:220].min(axis=2) > 200).sum()) > GAME_OVER_WHITE


def is_menu(frame):
    f = frame[10:100].astype(int)
    return int(((f[..., 0] > 200) & (f[..., 1] > 120) & (f[..., 1] < 210) & (f[..., 2] < 80)).sum()) > MENU_LOGO


def mario_candidates(frame):
    """Every blob that looks like Mario: [(x, feet_row), ...]. Mario = a small blob with both his bright red (cap,
    shirt: red > 180, almost no green/blue) and his dark navy overalls (blue 70-160, much bluer than red/green). The
    pinkish girders have too much blue for "red"; the oil drum is too bright a blue for "navy". The top strip (score,
    lives = little Marios, hammer/item icons), Kong's corner and the BONUS box are skipped.
    Barrels (blue and orange) can also pass, so the env tracks Mario instead of trusting any single blob (_update_mario).
    v1b: at least MIN_NAVY navy and MIN_RED red pixels and Mario's size. Measured: Mario 76-129 navy, 60-100 red,
    17-28 px wide, 20-28 tall; the oil-drum fire ~9 navy, a fireball (blue flame, red eyes) ~28 navy. In v1 the
    tracker drifted onto a fireball and paid a fake climb to line 6."""
    f = frame.astype(int)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    navy = (b >= 70) & (b < 160) & (b - r >= 50) & (b - g >= 50)
    red = (r > 180) & (g < 45) & (b < 45)
    skip = np.zeros(navy.shape, bool)
    skip[:58, :] = True          # HUD: score, lives, items
    skip[:112, :115] = True      # HUD icons + Kong's corner
    skip[:100, 380:] = True      # BONUS box
    navy &= ~skip
    red &= ~skip
    n, _, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate((navy | red).astype(np.uint8), np.ones((3, 3), np.uint8)))
    out = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        nv, rd = int(navy[y:y + h, x:x + w].sum()), int(red[y:y + h, x:x + w].sum())
        if nv >= MIN_NAVY and rd >= MIN_RED and 14 <= w <= 40 and 16 <= h <= 40:
            out.append((int(x + w // 2), int(y + h - 2)))   # plain ints (numpy ints can't be saved as JSON)
    return out


def find_mario(frame, last=None, max_dist=None):
    """Mario's (x, feet_row) or None. With last=(x, feet) only blobs within max_dist pixels count (the closest wins);
    without it (a new life), only blobs on the bottom girder where Mario starts (feet below SPAWN_ROW)."""
    cands = mario_candidates(frame)
    if last is None:
        cands = [c for c in cands if c[1] >= SPAWN_ROW]
        return max(cands, key=lambda c: c[1]) if cands else None
    near = [(abs(c[0] - last[0]) + abs(c[1] - last[1]), c) for c in cands]
    near = [(d, c) for d, c in near if d <= max_dist]
    return min(near)[1] if near else None


class DonkeyKongEnv(gym.Env):
    def __init__(self, headless=True, record_file=None, record_every=50, games=1, game=1, log_all=True,
                 shared=False, best_file=None):
        """record_file: episodes for viewer.html (runs/<run>_episodes.jsonl): pictures for every record_every-th
        episode, a result line for all (log_all=False: only the saved episodes). In training all games write ONE file
        (shared=True, game = which game this is).
        best_file: shared by all games (runs/<run>_best.json): what any game has reached so far. Episodes worth keeping
        are saved with pictures whatever record_every says, with "why": the first time any game reaches a new line,
        the first win, and every new best episode ("best so far", by reward)."""
        super().__init__()
        self.best_file, self.game, self.games = best_file, game, games
        self.train_step = None       # the real training step (set by train.py before each batch, see set_train_step)
        self.recorder = ReplayRecorder(record_file, record_every, games, frame_size=(265, 245),   # half size
                                       meta={"game": "donkey_kong", "actions": ACTION_NAMES},
                                       keep=self._why_keep if best_file else None, log_all=log_all, shared=shared
                                       ) if record_file else None
        self.headless = headless
        self._start_browser()

        self.action_space = gym.spaces.Discrete(len(ACTIONS))
        # v2: 140x140 grayscale of the play area (OBS_CROP; Mario is ~9 px tall); train.py stacks 4.
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(OBS_SIZE, OBS_SIZE, 1), dtype=np.uint8)

        self.frame = None
        self.steps = 0
        self.in_life = False         # Mario seen in play in this episode
        self.best_line = 0           # highest checkpoint line reached in this life
        self.x, self.feet = None, None   # Mario's last trusted position (feet row)
        self.lost_steps = 0          # steps since the last trusted detection
        self.visited = set()         # exploration cells visited in this life
        self.ladders_reached = set() # v2: working ladders whose foot Mario reached in this life
        self.approach = None         # v2b: (girder, target ladder x, distance) at the last step, or None
        self.recent_feet = []        # Mario's last GROUND_STEPS feet rows (for his standing height)
        self.total_reward = 0.0

    # ---- gym API -----------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        try:
            self.browser.hold([])
            if not self.in_life and time.time() - self.browser_started > RESTART_BROWSER_MINUTES * 60:
                self.browser.close()              # Mario just died: a fresh browser (see RESTART_BROWSER_MINUTES)
                self._start_browser()
            frame = self.screenshot()
        except Exception as e:                    # the page crashed: start a fresh browser
            print(f"browser failed in reset ({type(e).__name__}), restarting it", flush=True)
            self._restart_after_crash()
            frame = self.screenshot()
        self.frame = frame
        self.steps = 0
        self.total_reward = 0.0
        # Mario may still be alive (time limit, or a reset after a PPO update): then this episode starts where he is,
        # keeps tracking him, and counts the lines he is already above as reached (no free reward for them).
        if self.in_life and is_playing(frame):
            self.best_line = int(max(0, (START_ROW - self.feet) // CHECKPOINT_SPACING))   # same life: cells kept
        else:
            self.in_life, self.x, self.feet, self.best_line = False, None, None, 0
            self.visited, self.recent_feet, self.ladders_reached, self.approach = set(), [], set(), None
        self.lost_steps = 0
        if self.recorder:
            if self.recorder.active and self.recorder.episode_steps:
                self.recorder.end_episode(self._result("cut"))
            self.recorder.start_episode()
        return self._get_obs(frame), {}

    def step(self, action):
        playing_before = self.in_life
        try:
            self.browser.hold(ACTIONS[action] if self.in_life else [])   # waiting screens: no keys
            time.sleep(STEP_SECONDS)
            frame = self.screenshot()
        except Exception as e:                             # the page crashed ("Target crashed"): seen once in a test
            return self._recover(e)
        self.steps += 1
        if self.train_step is not None:
            self.train_step += self.games                  # all games step together

        reward, died, won = 0.0, False, False
        if self.in_life and not is_playing(frame):
            # The stage is gone: Mario died (the "25 m" or Game Over screen came up) -- or reached Pauline (the stage
            # ends with a cut scene). Told apart by where Mario was last seen.
            if self.feet is not None and self.feet <= WIN_ROW:
                won = True
                reward += WIN_REWARD + WIN_TIME_BONUS * max(0.0, 1 - self.steps / MAX_STEPS)
            else:
                died = True
                reward -= DEATH_PENALTY
        else:
            if not self.in_life:
                self._handle_menus(frame)                  # Game Over -> Retry, main menu -> New Game
            reward += self._update_mario(frame, reward_lines=True)

        if playing_before and not (died or won):
            reward -= STEP_PENALTY                         # v2c: time costs while a life is being played
        terminated = died or won
        truncated = not terminated and self.steps >= MAX_STEPS
        info = {"height": self.best_line, "cells": len(self.visited), "ladders": len(self.ladders_reached),
                "died": died, "won": won, "playing": playing_before}

        self.total_reward += reward
        if self.recorder:
            if playing_before or terminated or truncated:      # no pictures of the waiting screens between lives
                self.recorder.add_step(frame, {"a": int(action), "r": round(float(reward), 3), "h": self.best_line,
                                               "c": len(self.visited),
                                               "f": -1 if self.feet is None else int(self.feet)})
            if terminated or truncated:
                self.recorder.end_episode(self._result("won" if won else "died" if died else "time limit"))
        if terminated:
            self.in_life, self.x, self.feet = False, None, None
            self.visited, self.recent_feet, self.ladders_reached, self.approach = set(), [], set(), None
        self.frame = frame
        return self._get_obs(frame), reward, terminated, truncated, info

    # ---- helpers -----------------------------------------------------------
    def _restart_after_crash(self):
        try:
            self.browser.close()
        except Exception:
            pass
        self._start_browser()
        self.in_life, self.x, self.feet = False, None, None
        self.visited, self.recent_feet, self.ladders_reached, self.approach = set(), [], set(), None

    def _recover(self, error):
        """The browser crashed mid-episode: restart it and end this episode as cut off (truncated, no penalty), so one
        crashed game never stops a training run."""
        print(f"browser failed ({type(error).__name__}: {str(error)[:80]}), restarting it", flush=True)
        self._restart_after_crash()
        frame = self.screenshot()
        self.frame = frame
        if self.recorder:
            self.recorder.end_episode(self._result("crash"))
        info = {"height": self.best_line, "cells": len(self.visited), "ladders": len(self.ladders_reached),
                "died": False, "won": False, "playing": False}
        return self._get_obs(frame), 0.0, False, True, info

    def _start_browser(self):
        """Open the game and start a new game (main menu -> Enter; the intro then takes ~14 s, played as waiting steps)."""
        self.browser = self._open_game(self.headless)
        prepare_page(self.browser)
        tap(self.browser, "Enter")
        self.last_enter = time.time()
        self.browser_started = time.time()

    @staticmethod
    def _open_game(headless, tries=4):
        """Open the page (retried: with 10 games starting together, one page load timed out after 30 s once,
        probably the page's ads, and that crashed the whole training)."""
        for attempt in range(tries):
            try:
                browser = Browser(GAME_URL, headless=headless, viewport=VIEWPORT, extra_args=BLOCK_ADS)
                time.sleep(LOAD_SECONDS)
                return browser
            except Exception as e:                          # e.g. playwright TimeoutError on page.goto
                print(f"opening the game failed ({type(e).__name__}), try {attempt + 1} of {tries}", flush=True)
                try:
                    browser.close()
                except Exception:
                    pass
                time.sleep(5)
        raise RuntimeError("could not open the game")

    def _update_mario(self, frame, reward_lines):
        """Track Mario's height; return the checkpoint reward for new lines (once per life)."""
        if not is_playing(frame):
            return 0.0
        if self.feet is None:
            pos = find_mario(frame)                        # new life: only on the bottom girder
        else:
            pos = find_mario(frame, (self.x, self.feet), min(TRACK_MAX, TRACK_DIST + TRACK_GROW * self.lost_steps))
            rise_limit = min(MAX_RISE_LOST, MAX_RISE + CLIMB_PER_STEP * self.lost_steps)
            if pos is not None and self.feet - pos[1] > rise_limit:
                pos = None                                 # too far up for Mario: something else (v1: a fireball)
            if pos is None and self.lost_steps >= REACQUIRE_AFTER:
                # lost for a while (e.g. he fell off a girder): take a Mario-like blob anywhere, but not higher up
                low = [c for c in mario_candidates(frame) if c[1] >= self.feet - MAX_RISE]
                pos = min(low, key=lambda c: abs(c[0] - self.x) + abs(c[1] - self.feet)) if low else None
        if pos is None:
            self.lost_steps += 1
            return 0.0
        self.lost_steps = 0
        self.x, self.feet = pos
        self.recent_feet = (self.recent_feet + [self.feet])[-GROUND_STEPS:]
        cell = (self.x // EXPLORE_CELL[0], max(self.recent_feet) // EXPLORE_CELL[1])   # standing height, not jumps
        line = int(max(0, (START_ROW - self.feet) // CHECKPOINT_SPACING))
        if not self.in_life:
            self.in_life = True                            # a new life (or the episode started mid-life)
            self.best_line = line
            self.visited = {cell}
            return 0.0
        reward = 0.0
        if line > self.best_line:
            reward += (line - self.best_line) * CHECKPOINT_REWARD
            self.best_line = line
        if cell not in self.visited:
            self.visited.add(cell)
            reward += EXPLORE_REWARD
        standing = max(self.recent_feet)
        for i, (lx, foot) in enumerate(LADDERS):
            if i not in self.ladders_reached and abs(self.x - lx) <= LADDER_X_TOL and abs(standing - foot) <= LADDER_Y_TOL:
                self.ladders_reached.add(i)
                reward += LADDER_REWARD
        reward += self._approach_reward(standing)
        return reward if reward_lines else 0.0

    def _approach_reward(self, standing):
        """v2b: reward for getting closer to the nearest working ladder up from Mario's girder (see LADDER_APPROACH)."""
        g = min(range(len(GIRDER_ROWS)), key=lambda i: abs(standing - GIRDER_ROWS[i]))
        if abs(standing - GIRDER_ROWS[g]) > GIRDER_TOL:
            self.approach = None                           # between girders (on a ladder): no target
            return 0.0
        target = min((lx for (lx, _), lg in zip(LADDERS, LADDER_GIRDER) if lg == g), key=lambda lx: abs(self.x - lx))
        dist = abs(self.x - target)
        reward = 0.0
        if self.approach is not None and self.approach[:2] == (g, target):
            reward = LADDER_APPROACH * (self.approach[2] - dist) / APPROACH_SCALE
        self.approach = (g, target, dist)
        return reward

    def set_train_step(self, step):
        """The real training step (train.py calls this before every batch), so saved episodes carry it."""
        self.train_step = int(step)

    def _result(self, ended):
        """The result line of the episode that just ended (for the recorder / viewer)."""
        r = {"ended": ended, "game": self.game, "height": self.best_line, "cells": len(self.visited),
             "ladders": len(self.ladders_reached), "reward": round(self.total_reward, 2), "length": self.steps}
        if self.train_step is not None:
            r["train_step"] = self.train_step
        return r

    def _why_keep(self, result):
        """Reasons to save this episode with pictures (called by the recorder): the first time any game reaches a new
        line (from MIN_RECORD_LINE up), the first win, a new best episode (highest reward so far). best_file is shared
        by all games."""
        try:
            with open(self.best_file, encoding="utf-8") as f:
                state = json.load(f)
        except (OSError, ValueError):
            state = {}                                     # no file yet (or half-written by another game)
        reasons = []
        line = result["height"]
        if line >= MIN_RECORD_LINE and line > state.get("line", 0):
            reasons.append(f"first time at line {line}")
            state["line"] = line
        if result["ended"] == "won" and not state.get("won"):
            reasons.append("first win")
            state["won"] = True
        if line >= MIN_RECORD_LINE and result["reward"] >= state.get("reward", float("-inf")) + BEST_MARGIN:
            reasons.append("best so far")
            state["reward"] = result["reward"]
        if reasons:
            try:
                with open(self.best_file, "w", encoding="utf-8") as f:
                    json.dump(state, f)
            except OSError:
                pass
        return reasons

    def _handle_menus(self, frame):
        """While waiting for the next life: press Enter on the Game Over screen (Retry is selected) or the main menu
        (New Game is selected). Not more often than RETRY_COOLDOWN."""
        if time.time() - self.last_enter < RETRY_COOLDOWN:
            return
        if is_game_over(frame) or is_menu(frame):
            tap(self.browser, "Enter")
            self.last_enter = time.time()

    def _get_obs(self, frame):
        x0, y0, x1, y1 = OBS_CROP
        img = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (OBS_SIZE, OBS_SIZE), interpolation=cv2.INTER_AREA)
        return img[:, :, None]

    def screenshot(self):
        """The game as an RGB numpy array (VIEWPORT height x width x 3)."""
        return self.browser.screenshot(GAME_SELECTOR)

    def release_keys(self):
        """Let go of all keys (while PPO updates the network, see train.py)."""
        self.browser.hold([])

    def close(self):
        if self.recorder:
            self.recorder.close()
        self.browser.close()
