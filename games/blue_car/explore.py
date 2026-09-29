"""Find out how Blue Car works from Python (checklist steps 3 and 4). No RL here.

    python games/blue_car/explore.py

Opens the game in a window, then:
  1. holds each entry of KEYS_TO_TRY for a few seconds (respawning in between) and saves a screenshot
     before/after each one to screenshots/, so you can see which keys steer, brake, etc.
  2. checks whether Backspace restarts the game when you haven't lost
  3. measures how many steps per second are possible with a screenshot every step
Look at the images in games/blue_car/screenshots/ afterwards and write what you find in NOTES.md.
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))   # screenshots/ below is relative to this game's folder

import time
from pathlib import Path

import cv2

from blue_car_env import GAME_URL, LOAD_SECONDS, RESPAWN_KEY
from core.browser import Browser

HEADLESS = False          # True = no window (faster, but you can't watch)
HOLD_SECONDS = 2
KEYS_TO_TRY = [["w"], ["s"], ["a"], ["d"], ["ArrowUp"], ["ArrowDown"], ["ArrowLeft"], ["ArrowRight"],
               ["w", "a"], ["w", "d"], ["Space"]]

OUT = Path("screenshots")
OUT.mkdir(exist_ok=True)


def save(name, frame):
    cv2.imwrite(str(OUT / f"{name}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


b = Browser(GAME_URL, headless=HEADLESS, viewport=(960, 600))
print(f"loading ({LOAD_SECONDS} s)...")
time.sleep(LOAD_SECONDS)
b.page.locator("canvas").click()
save("00_start", b.screenshot())

# 1. What does each key do?
for keys in KEYS_TO_TRY:
    name = "+".join(keys)
    b.press(RESPAWN_KEY)
    time.sleep(1)
    save(f"{name}_before", b.screenshot())
    b.hold(keys)
    time.sleep(HOLD_SECONDS)
    b.hold([])
    save(f"{name}_after", b.screenshot())
    print(f"held {name:18} -> screenshots/{name}_before.png / _after.png")

# 2. Does Backspace restart when you haven't lost? Drive a little, respawn, compare with the start.
b.press(RESPAWN_KEY); time.sleep(1)
b.hold(["w"]); time.sleep(1); b.hold([])
save("respawn_1_driven", b.screenshot())
b.press(RESPAWN_KEY); time.sleep(1)
save("respawn_2_after_backspace", b.screenshot())
print("respawn test -> screenshots/respawn_1_driven.png vs respawn_2_after_backspace.png "
      "(same as 00_start = Backspace always restarts)")

# 3. How fast can we go with a screenshot every step?
n = 30
start = time.perf_counter()
for _ in range(n):
    b.screenshot()
per_shot = (time.perf_counter() - start) / n
print(f"one screenshot takes {per_shot * 1000:.0f} ms -> at most {1 / per_shot:.0f} steps/s "
      f"(plus the time each action is held)")

b.close()
