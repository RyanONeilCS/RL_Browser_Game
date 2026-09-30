"""Record yourself driving Blue Car, for behavior cloning (the agent learns to copy you first).

    python games/blue_car/record_demo.py

A game window opens. Click into it once, then drive with WASD or the arrow keys, like a normal game.
Every step (same rhythm as the agent: ~0.15 s) this saves what the agent would see (the same 84x84 image
as blue_car_env._get_obs) and which action your keys match (the same ACTIONS list the agent uses).

- Backspace restarts (like in the game) and starts a new episode. The recorder never ends a run by itself:
  the detectors (lost / off road / cubes) were only calibrated on the start of the track and misfire in later,
  differently lit sections, so they're only logged here (for checking them), not acted on.
- Every 3rd step a full-size screenshot is saved to demos/<file name>_frames/ (for calibrating the detectors
  on the whole track).
- Stop with Ctrl+C in this terminal (or close the game window): everything recorded so far is saved to
  demos/blue_car_<date>_<time>.npz. Record as many sessions as you like; each gets its own file.
- Drive the way you want the agent to drive: stay on the road, go for the cubes, don't crash on purpose.
  A few clean runs past cube 2 are worth more than many sloppy ones (see README: Recording human play).

No Python key presses here: your keys go straight to the game, Python only watches and takes screenshots.
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))   # demos/ below is relative to this game's folder

import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import blue_car_env as m

FRAME_EVERY = 3   # save a full-size screenshot every this many steps

# Which keys count as which key (arrow keys work like WASD).
KEY_NAMES = {"w": "w", "arrowup": "w", "a": "a", "arrowleft": "a", "d": "d", "arrowright": "d",
             "s": "s", "arrowdown": "s"}

# Watch the keys you hold inside the page (capture phase: before the game sees them).
# __restarts counts Backspace presses, so a quick tap between two steps is never missed.
KEY_LISTENER = """
window.__held = new Set();
window.__restarts = 0;
window.addEventListener('keydown', e => { window.__held.add(e.key.toLowerCase());
                                          if (e.key === 'Backspace' && !e.repeat) window.__restarts++; }, true);
window.addEventListener('keyup', e => window.__held.delete(e.key.toLowerCase()), true);
window.addEventListener('blur', () => window.__held.clear());
"""


def keys_to_action(held):
    """The ACTIONS index that matches the held keys (reverse doesn't exist for the agent: counts as coast)."""
    keys = {KEY_NAMES[k] for k in held if k in KEY_NAMES}
    gas, left, right = "w" in keys, "a" in keys, "d" in keys
    if left and right:
        left = right = False   # both at once cancel out
    if gas and left:
        return 2
    if gas and right:
        return 3
    if gas:
        return 1
    if left:
        return 4
    if right:
        return 5
    return 0


def main():
    env = m.BlueCarEnv(headless=False)   # opens the window, waits for the game to load, clicks for focus
    b = env.browser
    b.js(KEY_LISTENER)
    print("Recording. Drive in the game window (click it first). Backspace = restart. Ctrl+C here = stop and save.")

    name = f"blue_car_{datetime.now():%Y%m%d_%H%M%S}"
    frames_dir = Path("demos") / f"{name}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    obs_list, actions, episode_starts, cubes_list = [], [], [], []
    det_lost, det_on_road, det_cube_px = [], [], []      # what the detectors said (for checking them only)
    prev = b.screenshot()
    new_episode, cubes, episodes = True, 0, 0
    try:
        while True:
            time.sleep(m.STEP_SECONDS)
            held, restarts = b.js("[Array.from(window.__held), window.__restarts]")
            frame = b.screenshot()

            # Backspace (pressed since the last step): the game restarts, the next step starts a new episode.
            if restarts or "backspace" in held:
                b.js("window.__restarts = 0")
                if not new_episode:
                    print(f"  episode {episodes}: {len(obs_list)} steps recorded so far, cubes counted {cubes}")
                new_episode, cubes = True, 0
                while "backspace" in b.js("Array.from(window.__held)"):   # wait until you let go
                    time.sleep(0.05)
                time.sleep(m.RESPAWN_SECONDS)
                prev = b.screenshot()
                continue

            step = len(obs_list)
            obs_list.append(env._get_obs(prev))          # what the agent saw ...
            actions.append(keys_to_action(held))         # ... and what you did
            episode_starts.append(new_episode)
            cubes += env._cube_collected(prev, frame)
            cubes_list.append(cubes)
            det_lost.append(env._is_game_over(frame))
            det_on_road.append(env._on_road(frame))
            det_cube_px.append(int(env._cube_mask(frame).sum()))
            if step % FRAME_EVERY == 0:
                cv2.imwrite(str(frames_dir / f"{step:05d}.jpg"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
            if new_episode:
                episodes += 1
                new_episode = False
            prev = frame
    except KeyboardInterrupt:
        pass
    except Exception as e:                               # e.g. the window was closed
        print("stopped:", type(e).__name__)
    finally:
        if obs_list:
            Path("demos").mkdir(exist_ok=True)
            path = f"demos/{name}.npz"
            np.savez_compressed(path, obs=np.array(obs_list), actions=np.array(actions),
                                episode_starts=np.array(episode_starts), cubes=np.array(cubes_list),
                                det_lost=np.array(det_lost), det_on_road=np.array(det_on_road),
                                det_cube_px=np.array(det_cube_px))
            counts = np.bincount(actions, minlength=len(m.ACTIONS))
            print(f"saved {len(obs_list)} steps from {episodes} episode(s) to {path} (+ screenshots in {frames_dir})")
            print("actions used:", {i: int(c) for i, c in enumerate(counts)}, "(0 coast, 1 gas, 2 gas+L, 3 gas+R, 4 L, 5 R)")
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
