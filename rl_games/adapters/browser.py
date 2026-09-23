"""Playwright (Chromium) adapter for games that run in a web page."""
from __future__ import annotations

import time
from typing import Any, Sequence

import cv2
import numpy as np
from playwright.sync_api import Frame, Page, sync_playwright

from rl_games.core.adapter import Adapter


class BrowserAdapter(Adapter):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._pw = None
        self._browser = None
        self.page: Page | None = None
        self.target: Page | Frame | None = None   # page or iframe the game lives in
        self._held: set[str] = set()

    # ---- lifecycle -------------------------------------------------------
    def start(self) -> None:
        cfg = self.config
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(
            headless=cfg["headless"],
            # Keep the game running at full speed when the window isn't focused.
            args=["--disable-background-timer-throttling",
                  "--disable-renderer-backgrounding",
                  "--disable-backgrounding-occluded-windows"],
        )
        self.page = self._browser.new_page(viewport=cfg["viewport"])
        self._load()

    def _load(self) -> None:
        cfg = self.config
        timeout = cfg["load_timeout_ms"]
        self.page.goto(cfg["url"], timeout=timeout)

        for sel in cfg["click_before_start"]:
            self.page.click(sel, timeout=timeout)

        if cfg["frame_selector"]:
            handle = self.page.wait_for_selector(cfg["frame_selector"], timeout=timeout)
            frame = handle.content_frame()
            if frame is None:
                raise RuntimeError(f"{cfg['frame_selector']} is not an iframe")
            self.target = frame
        else:
            self.target = self.page

        self.canvas = self.target.locator(cfg["canvas_selector"]).first
        self.canvas.wait_for(timeout=timeout)
        if cfg["ready_expr"]:
            self.target.wait_for_function(cfg["ready_expr"], timeout=timeout)
        for stmt in cfg["setup_exprs"]:
            self._eval(stmt)
        if cfg.get("click_canvas_on_start", True):
            self.canvas.click()   # gives the game keyboard focus
        self._held.clear()

    def close(self) -> None:
        try:
            if self._browser:
                self._browser.close()
        finally:
            if self._pw:
                self._pw.stop()
        self._browser = self._pw = self.page = self.target = None

    # ---- game I/O --------------------------------------------------------
    def reset(self, seed: int | None = None) -> dict[str, Any] | None:
        self._release_all()
        if self.config["reset_expr"]:
            self._eval(self.config["reset_expr"], seed=seed)
        else:
            self._load()   # no reset hook: reloading the page is the only way to start over
        return self.get_state()

    def step(self, keys: Sequence[str], frames: int | None = None) -> dict[str, Any] | None:
        frames = self.config["frames_per_action"] if frames is None else frames
        keys = list(keys)
        if self.config["timing"] == "stepped":
            return self._eval(self.config["step_expr"], keys=keys, frames=frames)
        self.hold(keys)
        time.sleep(frames / self.config["fps"])
        return self.get_state()

    def hold(self, keys: Sequence[str]) -> None:
        """Real-time input: hold exactly `keys`, releasing anything else that is down."""
        want = set(keys)
        for k in self._held - want:
            self.page.keyboard.up(k)
        for k in want - self._held:
            self.page.keyboard.down(k)
        self._held = want

    def _release_all(self) -> None:
        self.hold([])

    def get_state(self) -> dict[str, Any] | None:
        expr = self.config["state_expr"]
        return self._eval(expr) if expr else None

    def get_frame(self) -> np.ndarray:
        png = self.canvas.screenshot(type="png")
        bgr = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def eval_js(self, expr: str, **args: Any) -> Any:
        """Escape hatch: evaluate any JS expression in the game's page/frame."""
        return self._eval(expr, **args)

    def _eval(self, expr: str, **args: Any) -> Any:
        # Wrap as a function so config expressions can use keys/frames/seed by name.
        names = ", ".join(args) if args else ""
        return self.target.evaluate(f"({{{names}}}) => ({expr})", args)
