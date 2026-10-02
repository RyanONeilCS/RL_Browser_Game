"""General browser control for any game: open a page, hold keys, take screenshots, run JavaScript.

Game-specific code (URLs, which keys, how to read reward / game over) lives in games/<game>/, not here.

    from core.browser import Browser
    b = Browser("https://example.com/game", headless=False)
    b.hold(["ArrowUp"])          # keys stay down until the next hold()
    frame = b.screenshot()       # RGB numpy array
    b.close()

Frame control (optional, Browser(..., frame_control=True)): the game only moves when we say so, frame by frame,
as fast as the computer can instead of in real time. Each frame = advance the page clock by 1/60 s (DevTools virtual
time) + draw exactly one frame (HeadlessExperimental.beginFrame). Load with b.wait(seconds) (draws frames in real time),
then b.freeze(); after that b.advance(ms, screenshot=True) runs ms of game time and returns the last frame.
What was tried on Blue Car (Unity WebGL), 2026-10-01:
- Playwright's page.clock: after pausing it the car never moved again. Doesn't work for Unity.
- Virtual time alone: the game drew only 1 frame per advance, so steering barely worked (physics in one big frame).
- beginFrame alone: 60 frames per game-second, but the page clock ran in real time, so each frame looked ~3 ms long.
- Both together: clock exact (1000 ms per 60 frames), steering like real time, ~3.4x faster than real time (1 game).
"""
import base64
import threading
import time

import cv2
import numpy as np
from playwright.sync_api import sync_playwright

# Playwright's sync API allows only one instance per thread, so every Browser in this process
# shares it (each one still gets its own browser window). Stopped when the last one closes.
_playwright = None
_open = 0


class Browser:
    def __init__(self, url, headless=True, viewport=(700, 700), gpu=True, frame_control=False):
        """gpu=True: render with the real graphics card. Headless Chromium otherwise draws WebGL games on the
        CPU (SwiftShader), which made Blue Car screenshots ~35x slower (2.4 s instead of 67 ms).
        frame_control=True: nothing is drawn unless we ask (see the top of this file). Headless only."""
        global _playwright, _open
        if _playwright is None:
            _playwright = sync_playwright().start()
        _open += 1
        # keep the game running at full speed when the window is hidden or in the background
        args = ["--disable-background-timer-throttling",
                "--disable-renderer-backgrounding",
                "--disable-backgrounding-occluded-windows"]
        if gpu:
            args += ["--enable-gpu", "--use-angle=d3d11", "--ignore-gpu-blocklist"]   # d3d11 = Windows graphics
        self.frame_control = frame_control
        if frame_control:
            args += ["--deterministic-mode", "--enable-begin-frame-control", "--run-all-compositor-stages-before-draw"]
        self._browser = _playwright.chromium.launch(headless=headless, args=args)
        self.page = self._browser.new_page(viewport={"width": viewport[0], "height": viewport[1]})
        self._held = set()
        self._frozen = False
        if frame_control:
            self._cdp = self.page.context.new_cdp_session(self.page)
            self._expired = threading.Event()
            self._cdp.on("Emulation.virtualTimeBudgetExpired", lambda _: self._expired.set())
            self._tick = time.perf_counter() * 1000
            self.page.goto(url, wait_until="commit")      # "load" would wait forever: nothing is drawn by itself
        else:
            self.page.goto(url)

    def js(self, code, arg=None):
        """Run JavaScript in the page and return the result (for games that expose an API)."""
        return self.page.evaluate(code, arg)

    def hold(self, keys):
        """Hold exactly these keys (e.g. ["ArrowUp", "ArrowLeft"]); keys not listed are released."""
        want = set(keys)
        for k in self._held - want:
            self.page.keyboard.up(k)
        for k in want - self._held:
            self.page.keyboard.down(k)
        self._held = want

    def press(self, key):
        """Tap a key once (e.g. "Enter" or "r" to restart)."""
        self.page.keyboard.press(key)

    # ---- frame control ----------------------------------------------------------------------------------
    FRAME_MS = 1000 / 60

    def wait(self, seconds):
        """Let the page run for this long in real time (frame control: keep drawing frames meanwhile)."""
        if not self.frame_control:
            time.sleep(seconds)
            return
        end = time.time() + seconds
        while time.time() < end:
            self._frame()
            time.sleep(self.FRAME_MS / 1000)

    def freeze(self, selector="canvas"):
        """Frame control: stop real time. From now on the game only moves with advance()."""
        box = self.page.evaluate(f"(() => {{ const r = document.querySelector('{selector}').getBoundingClientRect();"
                                 f" return [r.x, r.y, r.width, r.height]; }})()")
        self._canvas = [int(round(v)) for v in box]          # canvas position in the window, for screenshots
        self._cdp.send("Emulation.setVirtualTimePolicy", {"policy": "pause"})
        self._frozen = True

    def advance(self, ms, screenshot=False):
        """Frame control: run `ms` of game time (one frame per 1/60 s), as fast as possible. screenshot=True returns
        the last frame (the game canvas, RGB)."""
        n = max(1, round(ms / self.FRAME_MS))
        for i in range(n):
            img = self._frame(shot=screenshot and i == n - 1)
        return img if screenshot else None

    def _frame(self, shot=False):
        """Draw exactly one frame; when frozen, first move the page clock forward by one frame."""
        if self._frozen:
            self._expired.clear()
            self._cdp.send("Emulation.setVirtualTimePolicy", {"policy": "advance", "budget": self.FRAME_MS})
            start = time.time()
            while not self._expired.is_set():
                if time.time() - start > 10:
                    raise TimeoutError("virtual time did not advance")
                self.page.wait_for_timeout(1)     # lets Playwright deliver the "budget expired" event
        self._tick += self.FRAME_MS
        args = {"frameTimeTicks": self._tick, "interval": self.FRAME_MS, "noDisplayUpdates": False}
        if shot:
            args["screenshot"] = {"format": "jpeg", "quality": 90}
        r = self._cdp.send("HeadlessExperimental.beginFrame", args)
        if not shot:
            return None
        if not r.get("screenshotData"):           # nothing changed on screen: draw a forced frame
            return self._frame(shot=True) if self._frozen else None
        page = cv2.imdecode(np.frombuffer(base64.b64decode(r["screenshotData"]), np.uint8), cv2.IMREAD_COLOR)
        page = cv2.cvtColor(page, cv2.COLOR_BGR2RGB)
        # cut out the canvas (it can start above the window's top edge, e.g. y = -21: those rows stay black)
        x, y, w, h = self._canvas
        out = np.zeros((h, w, 3), np.uint8)
        top, left = max(0, -y), max(0, -x)
        src = page[max(0, y): max(0, y) + h - top, max(0, x): max(0, x) + w - left]
        out[top: top + src.shape[0], left: left + src.shape[1]] = src
        return out

    def screenshot(self, selector="canvas"):
        """Screenshot of `selector` (default: the game canvas; None = whole page) as an RGB numpy array."""
        if self.frame_control and self._frozen and selector == "canvas":
            return self._frame(shot=True)                        # one more frame (1/60 s), with the picture
        target = self.page.locator(selector).first if selector else self.page
        jpg = target.screenshot(type="jpeg", quality=90)   # JPEG is ~3x faster than PNG for WebGL games
        bgr = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def close(self):
        global _playwright, _open
        self._browser.close()
        _open -= 1
        if _open == 0:
            _playwright.stop()
            _playwright = None
