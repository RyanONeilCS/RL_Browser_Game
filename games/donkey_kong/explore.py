"""Find out how Donkey Kong (freekong.org) works from Python (checklist steps 3 and 4). No RL here.

    python games/donkey_kong/explore.py

Opens the game, starts a new game, then:
  1. saves a screenshot every second through the intro, to see when play really starts
  2. holds each entry of KEYS_TO_TRY and saves a screenshot before/after (which keys move, climb, jump)
  3. stands still and saves a screenshot every second until a life is lost / the game is over (barrels kill)
  4. measures how many steps per second are possible with a screenshot every step
Look at the images in games/donkey_kong/screenshots/ afterwards and write what you find in NOTES.md.
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))   # screenshots/ below is relative to this game's folder

import time
from pathlib import Path

import cv2

from donkey_kong_env import GAME_SELECTOR, GAME_URL, LOAD_SECONDS, VIEWPORT, prepare_page, tap
from core.browser import Browser

HEADLESS = False          # True = no window (faster, but you can't watch)
INTRO_SECONDS = 18        # screenshots every second for this long after pressing Enter
HOLD_SECONDS = 0.8
KEYS_TO_TRY = [["ArrowRight"], ["ArrowLeft"], ["ArrowUp"], ["ArrowDown"], ["Space"], ["ArrowRight", "Space"],
               ["z"], ["x"]]
WAIT_FOR_DEATH = 40       # stand still for up to this many seconds (one screenshot per second)

OUT = Path("screenshots")
OUT.mkdir(exist_ok=True)


def save(name, frame):
    cv2.imwrite(str(OUT / f"{name}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


b = Browser(GAME_URL, headless=HEADLESS, viewport=VIEWPORT)
print(f"loading ({LOAD_SECONDS} s)...")
time.sleep(LOAD_SECONDS)
prepare_page(b)
shot = lambda: b.screenshot(GAME_SELECTOR)
save("00_menu", shot())

# 1. Start a new game and watch the intro
tap(b, "Enter")
for i in range(INTRO_SECONDS):
    time.sleep(1)
    save(f"intro_{i:02d}", shot())
print(f"intro -> screenshots/intro_00.png .. intro_{INTRO_SECONDS - 1:02d}.png (when does play start?)")

# 2. What does each key do?
for keys in KEYS_TO_TRY:
    name = "+".join(keys)
    save(f"key_{name}_before", shot())
    b.hold(keys)
    time.sleep(HOLD_SECONDS)
    b.hold([])
    save(f"key_{name}_after", shot())
    print(f"held {name:18} -> screenshots/key_{name}_before.png / _after.png")

# 3. Stand still: how does dying look, what happens after (next life? game over? back to the menu?)
for i in range(WAIT_FOR_DEATH):
    time.sleep(1)
    save(f"still_{i:02d}", shot())
print(f"standing still -> screenshots/still_00.png .. still_{WAIT_FOR_DEATH - 1:02d}.png")

# 4. How fast can we go with a screenshot every step?
n = 30
start = time.perf_counter()
for _ in range(n):
    shot()
per_shot = (time.perf_counter() - start) / n
print(f"one screenshot takes {per_shot * 1000:.0f} ms -> at most {1 / per_shot:.0f} steps/s "
      f"(plus the time each action is held)")

b.close()
