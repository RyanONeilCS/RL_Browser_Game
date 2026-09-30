"""Record episodes for the replay viewer (e.g. games/blue_car/viewer.html), for any game.

One .jsonl file per run: a header line, then one line per finished episode with its result. Every
`every`-th episode also gets small JPEG frames and per-step data, so the viewer can play it back.

    rec = ReplayRecorder("runs/v6_replay.jsonl", every=5, steps_per_env_step=10, meta={"game": "blue_car"})
    rec.start_episode()
    rec.add_step(frame_rgb, {"a": action, "r": reward})     # every step
    rec.end_episode({"ended": "won", "cubes": 12})          # when the episode is over
"""
import base64
import json
import os
from pathlib import Path

import cv2


class ReplayRecorder:
    def __init__(self, path, every=5, steps_per_env_step=1, frame_size=(240, 144), jpeg_quality=70, meta=None):
        """every: record frames for every this-many-th episode (the others only get a result line).
        steps_per_env_step: games running at once, so a game's step count x this ~ the training step."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, "a", encoding="utf-8")
        self.every, self.scale = every, steps_per_env_step
        self.size, self.quality = frame_size, jpeg_quality
        self.episode, self.total_steps = 0, 0
        self.frames, self.data, self.active = [], [], False
        self.episode_steps = 0
        if os.path.getsize(path) == 0:
            self._write({"type": "run", "every": every, "frame_size": list(frame_size), **(meta or {})})

    def start_episode(self):
        self.frames, self.data = [], []
        self.active = True
        self.episode_steps = 0
        self.recording = self.episode % self.every == 0

    def add_step(self, frame_rgb, data):
        """frame_rgb: the (cropped) game picture; data: small per-step values (action, reward, ...)."""
        if not self.active:
            return
        self.total_steps += 1
        self.episode_steps += 1
        if self.recording:
            small = cv2.resize(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), self.size, interpolation=cv2.INTER_AREA)
            ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            self.frames.append(base64.b64encode(jpg.tobytes()).decode("ascii"))
            self.data.append(data)

    def end_episode(self, result):
        """result: how it ended (e.g. {"ended": "won", "cubes": 12, "reward": 45.2}). Writes one line."""
        if not self.active:
            return
        line = {"type": "episode", "episode": self.episode, "train_step": self.total_steps * self.scale,
                "steps": len(self.data) if self.recording else None, **result}
        if self.recording and self.frames:
            line["frames"], line["data"] = self.frames, self.data
        self._write(line)
        self.episode += 1
        self.active = False
        self.frames, self.data = [], []

    def close(self):
        self.file.close()

    def _write(self, obj):
        self.file.write(json.dumps(obj, separators=(",", ":")) + "\n")
        self.file.flush()   # so the viewer can load the file while training is still running
