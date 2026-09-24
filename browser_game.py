"""Runs the toy car game in Chromium (via Playwright) so Python can play it.

You shouldn't need to change this file. Run it directly to watch random driving:

    python browser_game.py

Recording for viewer.html: pass record_file="runs/my_run.jsonl" and the car's path
for every `record_every`-th episode is saved while you train (works with any agent).
"""
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
from playwright.sync_api import sync_playwright

GAME_URL = (Path(__file__).parent / "game" / "index.html").resolve().as_uri()


class BrowserGame:
    def __init__(self, headless=False, difficulty="easy", realtime=False, frames_per_action=4,
                 record_file=None, record_every=10):
        """
        realtime=False: Python advances the game exactly `frames_per_action` frames per step
                        (deterministic and fast, use this for training).
        realtime=True:  the game runs on its own clock and Python holds keys like a human.
        record_file:    path of a .jsonl file to save episode paths to (for viewer.html), or None.
        """
        self.realtime = realtime
        self.frames_per_action = frames_per_action
        self._held = set()

        self._record = None
        self._record_every = record_every
        self._episode = 0          # finished episodes
        self._total_steps = 0      # steps across all episodes
        self._path = []

        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(
            headless=headless,
            args=["--disable-background-timer-throttling",
                  "--disable-renderer-backgrounding",
                  "--disable-backgrounding-occluded-windows"],
        )
        self.page = self._browser.new_page(viewport={"width": 700, "height": 700})
        self.page.goto(GAME_URL)
        self.page.wait_for_function("window.gameReady === true")
        self.page.evaluate(f"window.gameConfig.difficulty = '{difficulty}'")
        self.page.evaluate(f"window.setMode('{'realtime' if realtime else 'stepped'}')")
        self.canvas = self.page.locator("canvas")
        self.reset()

        if record_file:
            Path(record_file).parent.mkdir(parents=True, exist_ok=True)
            self._record = open(record_file, "w", encoding="utf-8")
            track = self.page.evaluate("window.getTrack()")
            self._write({"type": "track", "difficulty": difficulty, "recordEvery": record_every,
                         "halfWidth": track["halfWidth"], "gates": track["gates"],
                         "centerline": [[round(p["x"], 1), round(p["y"], 1)] for p in track["centerline"]]})

    def reset(self, seed=None):
        """New episode. An int seed gives a random start position on the track. Returns the state."""
        self._hold([])
        self._finish_episode()
        state = self.page.evaluate("s => window.resetGame(s ?? undefined)", seed)
        self._last_state = state
        return state

    def step(self, keys):
        """Hold `keys` (e.g. ["ArrowUp", "ArrowLeft"]) for one step. Returns the new state."""
        if not self.realtime:
            state = self.page.evaluate("([k, f]) => window.stepGame(k, f)", [list(keys), self.frames_per_action])
        else:
            self._hold(keys)
            time.sleep(self.frames_per_action / 60)
            state = self.state()
        self._total_steps += 1
        if self._record and self._episode % self._record_every == 0:
            if not self._path:
                s = self._last_state
                self._path.append([round(s["x"]), round(s["y"]), round(s["angle"], 2), round(s["speed"])])
            self._path.append([round(state["x"]), round(state["y"]), round(state["angle"], 2), round(state["speed"])])
        self._last_state = state
        return state

    def state(self):
        """The game's state dict: speed, rays, checkpoints, crashed, done, ... (see getState in game/game.js)."""
        return self.page.evaluate("window.gameState")

    def frame(self):
        """Screenshot of the game as an RGB numpy array (600, 600, 3)."""
        png = self.canvas.screenshot(type="png")
        bgr = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def close(self):
        self._finish_episode()
        if self._record:
            self._record.close()
        self._browser.close()
        self._pw.stop()

    def _finish_episode(self):
        # Called on reset/close: count the episode that just ended and save its path if recorded.
        if getattr(self, "_last_state", None) is None or self._last_state["frame"] == 0:
            return   # nothing happened since the last reset
        s = self._last_state
        if self._path:
            self._write({"type": "episode", "episode": self._episode, "totalSteps": self._total_steps,
                         "checkpoints": s["checkpointsPassed"], "numCheckpoints": s["numCheckpoints"],
                         "crashed": s["crashed"], "finished": s["done"] and not s["crashed"],
                         "seconds": round(s["frame"] * s["dt"], 2), "path": self._path})
        self._episode += 1
        self._path = []
        self._last_state = None

    def _write(self, obj):
        self._record.write(json.dumps(obj, separators=(",", ":")) + "\n")
        self._record.flush()   # so the viewer can load the file while training is still running

    def _hold(self, keys):
        # Real-time mode only: press new keys, release ones no longer wanted.
        if not self.realtime:
            return
        want = set(keys)
        for k in self._held - want:
            self.page.keyboard.up(k)
        for k in want - self._held:
            self.page.keyboard.down(k)
        self._held = want


if __name__ == "__main__":
    # Random driving, just to check everything works.
    actions = [[], ["ArrowUp"], ["ArrowLeft"], ["ArrowRight"],
               ["ArrowUp", "ArrowLeft"], ["ArrowUp", "ArrowRight"], ["ArrowDown"]]
    game = BrowserGame()
    state = game.reset()
    print("difficulty:", state["difficulty"])
    start = time.perf_counter()
    for i in range(300):
        state = game.step(random.choice(actions))
        if state["done"] or state["truncated"]:
            print(f"step {i}: crashed={state['crashed']} checkpoints={state['checkpointsPassed']}")
            state = game.reset(seed=i)
    print(f"{300 / (time.perf_counter() - start):.0f} steps/s")
    print("state keys:", sorted(state))
    game.close()
