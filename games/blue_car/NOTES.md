# Blue Car: notes

Everything found out about the game so far, and where it is on the
[new-game checklist](../../README.md#adding-a-new-game).

## The game

- itch.io page: https://kaboochy.itch.io/blue-car (title in the browser: "DerbyCar")
- Game URL (the iframe inside the itch page): https://html-classic.itch.zone/html/13523181/index.html
- Engine: **Unity WebGL**, draws to one `<canvas>`.
- ⚠️ The itch page says: "V0.009 … CURRENTLY NOT WORKING!!! PROBABLY JUST GOING TO POST ON STEAM!"
  It does load and drive (see below), but it may be buggy or change / disappear. Ask the developer for a
  web export and keep a local copy in `game/` so the URL can't break under you.
- 3D, night, low-poly: a road through a forest, camera behind the blue car, dashed centre line.

## Found so far (tested from Python, 2026-09-29)

| What | Finding |
|---|---|
| Loads headless? | Yes (screenshots work with `headless=True`). Needs ~15–20 s to load. |
| Focus | Click the canvas once before keys work. |
| Gas | `w` and `ArrowUp` both drive forward. |
| Steering | `a`/`d` and `ArrowLeft`/`ArrowRight` do nothing while standing still; `w+a` / `w+d` steer hard (off the road within 2 s of holding). |
| `s` / `ArrowDown` | Reversing **right at the start** → YOU LOSE. After driving forward first, reversing (also `s+a`) is fine. |
| Losing | Driving off the road or reversing → **"YOU LOSE / PRESS BACKSPACE TO RESPAWN"** in white text in the middle. |
| Restart | `Backspace` restarts **any time** (also without losing): back to the exact start. So `reset()` works. |
| Number bottom-left | **Cubes collected.** White digit on a dark background: when it changes, ~1,200 of its white pixels flip (grain: at most ~10). Respawn resets it to 0 within 0.1 s. |
| Off the road | Doesn't lose immediately: after 2 s in the grass/trees there was no YOU LOSE yet (it came after ~3–12 s). |
| Game-over detector | White pixels (all colours > 200) in rows 250–350, cols 100–860: lose screen ~11,450, all 23 normal screenshots 0. Threshold 2000. |
| Orange cube on the road | Probably a pickup counted by the number (see above). |
| Sound | Browser blocks audio until a click ("AudioContext was not allowed to start"): harmless. |
| Screenshot speed | Headless default rendered on the CPU: **2.4 s** per screenshot. With GPU flags (now the default in `core/browser.py`) and JPEG: **~67 ms**. |
| Steps per second | **~6.7** for one game (20k steps ≈ 50 min, 500k ≈ 21 h). |
| Several games at once | 4 games 25 steps/s, 8 → 47, **12 → 62**, 16 → 73 (but only 2.6 GB RAM left). Training uses 12. |
| PPO update pause | ~4.5 s per 3k steps on the CPU: the games keep running meanwhile, so `core/realtime.py` releases keys and restarts all games around each update. |

Key test screenshots: run `explore.py` (saved in `screenshots/`, not in git).

## Checklist progress

- [x] 1. Investigate: engine, URL, controls, lose screen, restart key, the number (probably cubes)
- [x] 2. Folder created (`blue_car_env.py`, `explore.py`, `train.py`, `play.py`, `test.py`)
- [x] 3. Running from Python: loads, focus, keys, screenshots, headless, respawn (`explore.py`)
- [x] 4. Timing: ~6.7 steps/s in real time. Faster needs clock control (later) or a shorter screenshot.
- [x] 5. Actions: 7, like the toy car (coast, gas, gas+left/right, left/right, brake/reverse)
- [x] 6. Observations: crop, gray, contrast boost (`equalizeHist`), 84×84, 4 stacked frames (in train.py)
- [x] 7. Reward v1: +1 per cube, −10 on losing, −0.01 per step
- [x] 8. Done / reset: `_is_game_over` (white text count), Backspace respawn; time limit 500 steps
- [x] 9. `check_env` passes; random agent: episodes end correctly, but **0 cubes in 3 episodes** (exploration may be slow)
- [ ] 10. Train: v1 overnight, 12 games, 1.5M steps (~7.5 h)
- [ ] 11. Evaluate the checkpoints with `test.py`, compare with the random agent

## Things to watch in the v1 run

- `rollout/cubes` (average cubes per episode) is the number that matters; `ep_rew_mean` mixes cubes and penalties.
- If `cubes` stays ~0 for hundreds of thousands of steps, the agent never finds cubes by chance: the reward is
  too sparse (like the toy car's v2). Ideas then: a small reward for moving forward / staying on the road,
  or the "cube channel" observation so cubes are easier to see.
- The cubes are hard to see in the gray image (tiny far away, a white blob up close). Colour or a cube channel
  are the backup plans.

## Questions for the developer

- Controls? What is the number bottom-left, and what are the orange cubes?
- Is there an end to the road (finish line / lap) or does it go on forever?
- Can we get the web export (the files behind the itch page) to run locally?
- Does the game use the browser's clock for its timing (for speeding it up later)?
