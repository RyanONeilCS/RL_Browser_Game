"""Record episodes for the replay viewer (e.g. games/blue_car/viewer.html), for any game.

One .jsonl file per run: a header line, then one line per finished episode with its result. Every
`every`-th episode also gets small JPEG frames and per-step data, so the viewer can play it back.

    rec = ReplayRecorder("runs/v6_replay.jsonl", every=5, steps_per_env_step=10, meta={"game": "blue_car"})
    rec.start_episode()
    rec.add_step(frame_rgb, {"a": action, "r": reward})     # every step
    rec.end_episode({"ended": "won", "cubes": 12})          # when the episode is over

Optional, for several games writing ONE file (Donkey Kong):
  keep=function(result) -> list of reasons, e.g. ["first time at line 4"]: any episode it gives a reason for is also
      saved with its pictures (so every episode's frames are kept in memory until it ends). The saved line gets
      "why": the reasons, plus "every Nth" when it was also a scheduled one.
  log_all=False: no result line for the other episodes (only game 1 logs every result, for the viewer's chart).
  shared=True: several processes append to the same file; each write takes a lock (path + ".lock") first, so lines
      never get mixed up.
"""
import base64
import json
import os
import time
from pathlib import Path

import cv2

if os.name == "nt":
    import msvcrt
else:
    import fcntl


class ReplayRecorder:
    def __init__(self, path, every=5, steps_per_env_step=1, frame_size=(240, 144), jpeg_quality=70, meta=None,
                 keep=None, log_all=True, shared=False):
        """every: record frames for every this-many-th episode (the others only get a result line); 0 = none.
        steps_per_env_step: games running at once, so a game's step count x this ~ the training step.
        keep / log_all / shared: see the top of this file."""
        self.path, self.every, self.scale = path, every, steps_per_env_step
        self.keep, self.log_all, self.shared = keep, log_all, shared
        self.size, self.quality = frame_size, jpeg_quality
        self.episode, self.total_steps = 0, 0
        self.frames, self.data, self.active = [], [], False
        self.episode_steps = 0
        self.header = {"type": "run", "every": every, "frame_size": list(frame_size), **(meta or {})}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.file = None if shared else open(path, "a", encoding="utf-8")
        self._write(None)                                   # writes the header if the file is new

    def start_episode(self):
        self.frames, self.data = [], []
        self.active = True
        self.episode_steps = 0
        self.scheduled = self.log_all and bool(self.every) and self.episode % self.every == 0
        self.recording = self.scheduled or self.keep is not None   # with keep: decide at the end

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
        """result: how it ended (e.g. {"ended": "won", "cubes": 12, "reward": 45.2}). Writes one line (or none)."""
        if not self.active:
            return
        reasons = list(self.keep(result) or []) if self.keep else []
        save = (self.scheduled or reasons) and self.frames
        line = {"type": "episode", "episode": self.episode, "train_step": self.total_steps * self.scale, **result}
        if reasons or (self.scheduled and self.keep):
            line["why"] = ([f"every {self.every}th"] if self.scheduled else []) + reasons
        if save:
            line.update(steps=len(self.data), frames=self.frames, data=self.data)
        if save or self.log_all:
            self._write(line)
        self.episode += 1
        self.active = False
        self.frames, self.data = [], []

    def close(self):
        if self.file:
            self.file.close()

    def _write(self, obj):
        """Append one line (obj=None: only the header, if the file is still empty)."""
        try:
            if not self.shared:
                if self.file.tell() == 0 and os.path.getsize(self.path) == 0:
                    self.file.write(json.dumps(self.header, separators=(",", ":")) + "\n")
                if obj is not None:
                    self.file.write(json.dumps(obj, separators=(",", ":")) + "\n")
                self.file.flush()   # so the viewer can load the file while training is still running
                return
            with open(self.path + ".lock", "a+") as lock:
                _lock(lock)
                try:
                    with open(self.path, "a", encoding="utf-8") as f:
                        if f.tell() == 0:
                            f.write(json.dumps(self.header, separators=(",", ":")) + "\n")
                        if obj is not None:
                            f.write(json.dumps(obj, separators=(",", ":")) + "\n")
                finally:
                    _unlock(lock)
        except OSError as e:
            print(f"replay: could not write to {self.path}: {e}", flush=True)   # never stop training for a replay


def _lock(f):
    """Wait until this process holds the lock on f (an open lock file)."""
    if os.name != "nt":
        fcntl.flock(f, fcntl.LOCK_EX)
        return
    f.seek(0)
    while True:
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            return
        except OSError:
            time.sleep(0.02)


def _unlock(f):
    if os.name != "nt":
        fcntl.flock(f, fcntl.LOCK_UN)
        return
    f.seek(0)
    msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
