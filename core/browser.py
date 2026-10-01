"""General browser control for any game: open a page, hold keys, take screenshots, run JavaScript.

Game-specific code (URLs, which keys, how to read reward / game over) lives in games/<game>/, not here.

    from core.browser import Browser
    b = Browser("https://example.com/game", headless=False)
    b.hold(["ArrowUp"])          # keys stay down until the next hold()
    frame = b.screenshot()       # RGB numpy array
    b.close()

Virtual time (optional): b.use_virtual_time() freezes the game; b.advance(150) then runs exactly 150 ms of game time
as fast as the computer can (Chrome DevTools virtual time). Tested on Blue Car (Unity WebGL): the game is frozen in
between, physics behaves as in real time, advancing 150 ms takes ~3 ms. (Playwright's page.clock does NOT work for
Unity: after pausing it the car never moved again.) With virtual time on, screenshot() uses a DevTools capture of the
canvas: same picture, ~39 ms instead of ~84 ms.
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
    def __init__(self, url, headless=True, viewport=(700, 700), gpu=True):
        """gpu=True: render with the real graphics card. Headless Chromium otherwise draws WebGL games on the
        CPU (SwiftShader), which made Blue Car screenshots ~35x slower (2.4 s instead of 67 ms)."""
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
        self._browser = _playwright.chromium.launch(headless=headless, args=args)
        self.page = self._browser.new_page(viewport={"width": viewport[0], "height": viewport[1]})
        self.page.goto(url)
        self._held = set()
        self._cdp = None             # DevTools session, set by use_virtual_time()

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

    def use_virtual_time(self, selector="canvas"):
        """Freeze the page's time; from now on it only moves with advance(). Call after the game has loaded."""
        self._cdp = self.page.context.new_cdp_session(self.page)
        self._expired = threading.Event()
        self._cdp.on("Emulation.virtualTimeBudgetExpired", lambda _: self._expired.set())
        self._clip = self.page.locator(selector).first.bounding_box()       # where the canvas is, for screenshots
        self._cdp.send("Emulation.setVirtualTimePolicy", {"policy": "pause"})
        self.advance(50)

    def advance(self, ms, timeout=10.0):
        """Run exactly `ms` milliseconds of game time (virtual time only), as fast as the computer can."""
        self._expired.clear()
        self._cdp.send("Emulation.setVirtualTimePolicy", {"policy": "advance", "budget": ms})
        start = time.time()
        while not self._expired.is_set():
            if time.time() - start > timeout:
                raise TimeoutError(f"virtual time did not advance {ms} ms within {timeout} s")
            self.page.wait_for_timeout(1)     # lets Playwright deliver the "budget expired" event

    def screenshot(self, selector="canvas"):
        """Screenshot of `selector` (default: the game canvas; None = whole page) as an RGB numpy array."""
        if self._cdp is not None and selector == "canvas":    # virtual time: faster DevTools capture, same picture
            c = self._clip
            shot = self._cdp.send("Page.captureScreenshot", {"format": "jpeg", "quality": 90, "optimizeForSpeed": True,
                                  "clip": {"x": c["x"], "y": c["y"], "width": c["width"], "height": c["height"],
                                           "scale": 1}})
            bgr = cv2.imdecode(np.frombuffer(base64.b64decode(shot["data"]), np.uint8), cv2.IMREAD_COLOR)
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
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
