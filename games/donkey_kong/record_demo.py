"""Record yourself playing Donkey Kong, for behavior cloning (the agent copies you first, then improves with PPO).

    python games/donkey_kong/record_demo.py

A game window opens and a new game starts (intro ~14 s). Click into the window once, then play with the ARROW KEYS
and SPACE (jump), like normal. Python only watches: your keys go straight to the game.
Every step (~0.14 s, the agent's rhythm) it saves what the agent would see (the same 140x140 picture as
donkey_kong_env._get_obs) and which of the agent's 7 actions your keys match.

- Only steps while a life is being played are saved (not the "25 m" screen, Game Over, the intro).
- Steps where you press nothing are saved with action -1: the agent has no "do nothing" action, so train_bc.py
  leaves them out (they're kept so the picture history before your next move stays complete).
- Game Over: it presses Enter (Retry) for you, so you can just keep playing.
- Stop with Ctrl+C in this terminal (or close the game window): everything is saved to
  demos/donkey_kong_<date>_<time>.npz. Record as many sessions as you like; each gets its own file.
- Play the way you want the agent to play: climb, jump barrels, wait for a barrel to pass before a ladder.
  Getting past the 3rd and 4th girders is what the agent can't do yet (v2), so those runs are worth the most.
  A few clean runs beat many sloppy ones; some recoveries (backing off a ladder, waiting) help too.

v1 of this file written by Claude on 2026-10-08 (adapted from games/blue_car/record_demo.py).
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))   # demos/ below is relative to this game's folder

import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import donkey_kong_env as m

FRAME_EVERY = 5        # also save a full-size screenshot every this many recorded steps (for checking)

# Watch the keys you hold inside the page (capture phase: before the game sees them).
KEY_LISTENER = """
window.__held = new Set();
window.addEventListener('keydown', e => window.__held.add(e.key), true);
window.addEventListener('keyup', e => window.__held.delete(e.key), true);
window.addEventListener('blur', () => window.__held.clear());
"""


def keys_to_action(held):
    """The ACTIONS index your keys match, or -1 for no movement key.
    Up / down win (climbing), then jumps (with the direction held), then walking."""
    up, down = "ArrowUp" in held, "ArrowDown" in held
    left, right = "ArrowLeft" in held, "ArrowRight" in held
    jump = " " in held or "Space" in held
    if left and right:
        left = right = False                 # both at once cancel out
    if up:
        return 2
    if down:
        return 3
    if jump:
        return 5 if left else 6 if right else 4
    if left:
        return 0
    if right:
        return 1
    return -1


def main(fake_keys=None, max_seconds=None, headless=False):
    """fake_keys(step) -> keys: for testing only (holds those keys instead of a human playing)."""
    env = m.DonkeyKongEnv(headless=headless)      # opens the game, clicks it, starts a new game
    b = env.browser
    b.js(KEY_LISTENER)
    print("Recording. Click the game window, then play with the arrow keys and Space. Ctrl+C here = stop and save.")

    name = f"donkey_kong_{datetime.now():%Y%m%d_%H%M%S}"
    frames_dir = Path("demos") / f"{name}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    obs_list, actions, starts, heights = [], [], [], []
    prev = env.screenshot()
    lives, step, t0 = 0, 0, time.time()
    prev_alive = False                                              # was the previous step a recorded one?
    try:
        while max_seconds is None or time.time() - t0 < max_seconds:
            if fake_keys:
                b.hold(fake_keys(step))
            time.sleep(m.STEP_SECONDS)
            held = set(b.js("Array.from(window.__held)"))
            frame = env.screenshot()
            step += 1

            was_alive = env.in_life
            if env.in_life and not m.is_playing(frame):           # the life is over (or Pauline was reached)
                print(f"  life {lives}: best line {env.best_line}, {len(obs_list)} steps recorded so far")
                env.in_life, env.x, env.feet = False, None, None
                env.visited, env.recent_feet, env.ladders_reached = set(), [], set()
            elif not env.in_life:
                env._handle_menus(frame)                            # Game Over -> Retry, main menu -> New Game
                env._update_mario(frame, reward_lines=False)        # starts tracking when a new life begins
            else:
                env._update_mario(frame, reward_lines=False)

            if was_alive and env.in_life:                           # a step of a life being played
                obs_list.append(env._get_obs(prev))                 # what the agent saw ...
                actions.append(keys_to_action(held))                # ... and what you did
                starts.append(len(starts) == 0 or not prev_alive)
                heights.append(env.best_line)
                if len(obs_list) % FRAME_EVERY == 0:
                    cv2.imwrite(str(frames_dir / f"{len(obs_list):05d}.jpg"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                                [cv2.IMWRITE_JPEG_QUALITY, 85])
            elif env.in_life and not was_alive:
                lives += 1
            prev_alive = was_alive and env.in_life
            prev = frame
    except KeyboardInterrupt:
        pass
    except Exception as e:                                          # e.g. the window was closed
        print("stopped:", type(e).__name__, str(e)[:80])
    finally:
        if obs_list:
            path = f"demos/{name}.npz"
            np.savez_compressed(path, obs=np.array(obs_list), actions=np.array(actions),
                                episode_starts=np.array(starts), heights=np.array(heights))
            acts = np.array(actions)
            counts = {m.ACTION_NAMES[i]: int((acts == i).sum()) for i in range(len(m.ACTIONS))}
            print(f"saved {len(obs_list)} steps from {sum(starts)} li(fe/ves) to {path} (best line {max(heights)})")
            print("actions:", counts, "| no key:", int((acts == -1).sum()))
        else:
            print("nothing recorded (no life was played)")
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
