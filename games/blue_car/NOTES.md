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
- [x] 10. Train v1 (overnight, 8 games, 660k steps in 6.9 h): **didn't learn.** From ~100k steps it stood still
      (episodes ran to the 500-step limit, reward ≈ −5, cubes ≈ 0): not moving never loses, so it beats driving
      and losing (−10). Needs reward v2 that pays for driving between cubes. Also slowed 45 → 26 steps/s from a
      memory leak (Python 3.14 + Playwright, see README Workflow notes); fixed by moving to Python 3.12.
- [x] 10b. Reward v2 (+ standing-still penalty 0.1 when the 84×84 picture changes < 20 per pixel), 250k steps:
      **learned to hide.** It drove off the road within ~2 s into a dark spot where the game never says YOU LOSE,
      and wiggled there for the whole episode (ep_len ~490, cubes 0, reward ~−6). The still penalty didn't
      catch it: in a near-black picture equalizeHist blows the grain up into "motion". Seen by running the
      250k checkpoint and saving frames (`screenshots/v2_250k_drive.png`).
      (This run also accidentally used the old Python 3.14 venv, so the memory leak came back.)
- [x] 10c. Reward v3: off the road for 10 steps in a row (no road markings in the lower screen) = lost (−10).
      Measured on 30 frames: on road 443–2,465 marking pixels, off road 0. Tested: the v2 hider is now
      caught after 25 steps.
- [x] 10d. v3 first hour (72k steps): **learning** (cubes 0.07 -> 0.34, ep_len 19 -> 31, no loophole), but only
      ~18 steps/s: episodes are short (~25 steps) and every reset waited 1 s, and the 10 games step in lockstep,
      so each reset stalled all of them. Respawn wait cut to 0.2 s (measured: done in < 0.1 s). Stopped and
      resumed from the newest checkpoint with the new `RESUME` setting in train.py.
- [x] 10e. v3 at 650k: drives on the road and collects the first cube almost every episode (cubes ~1.0),
      but never a second one: it dawdles, brakes a lot, sometimes ends up sideways at the road edge.
      Found: in darker sections (and sideways) the road only gives 110-190 marking pixels, so ROAD_THRESHOLD
      150 ended episodes as "off road" on the asphalt, punishing it for driving on. Lowered to 60 (real off
      road is always 0) and resumed from the 650k checkpoint.
      Open question: how far is the second cube? (Holding w reaches cube 1 in ~2 s; the agent takes 8-17 s.)
- [x] 10f. v3b (from v3 850k, ent_coef 0.01, episodes/batches 1000/1024 steps), 930k steps / 4.7 h: **didn't
      help.** More exploring (entropy −0.53 → −0.81) mostly meant leaving the road (off_road 54% → 98%,
      ep_len 464 → 214). Cube 2 reached in 1 of 92 batches: reachable, but far too rare to learn from.
      Also tried: optical flow of the road to measure forward movement: no separation between driving,
      standing and reversing (film grain + look-alike dashes).
- [x] 10g. Reward v4: +0.05 per step with gas on the road (reward the action, not a measurement),
      losing / off road −20 (was −10), reverse removed (action 6 = coast). Resumed from v3 800k, ent_coef 0.005.
      Checked before training: the v3 800k model scores −103.8 over a 1000-step episode under v4 (on the road
      but nearly still the whole time), standing still −32.7 per 300 steps, gas straight −18.4 (cube 1, then
      YOU LOSE at the first curve).
      From playing it (Ryan): cube 2 is only ~5 s in at normal driving (cube 1 ~2 s), so the agent (8–17 s to
      cube 1) is simply far too slow. A 180° turn isn't possible without leaving the track.
- [ ] 11. Train v4, then evaluate the checkpoints with `test.py`, compare with the random agent
- Next if v4 doesn't reach cube 2: learn from human demos (behavior cloning), or a progress value from
  the game's code (ask the developer).

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
