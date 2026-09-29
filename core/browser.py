"""General browser control for any game: open a page, hold keys, take screenshots, run JavaScript.

Game-specific code (URLs, which keys, how to read reward / game over) lives in games/<game>/, not here.

    from core.browser import Browser
    b = Browser("https://example.com/game", headless=False)
    b.hold(["ArrowUp"])          # keys stay down until the next hold()
    frame = b.screenshot()       # RGB numpy array
    b.close()
"""
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

    def screenshot(self, selector="canvas"):
        """Screenshot of `selector` (default: the game canvas; None = whole page) as an RGB numpy array."""
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
