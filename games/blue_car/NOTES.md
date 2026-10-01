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
- [x] 11. v4 (from v3 800k), 1.2M steps / 6.2 h: **worse.** off_road 79% -> 98-100% within ~100k steps,
      cubes 0.84 -> 0.48, cube 2 in 2 of 118 batches. Cause (checked by running v4_final and splitting the reward):
      the still check fires while driving: 39% of gas-on-road steps counted as "still" (picture change median 23,
      threshold 20, measured only at the bright start). Over 4 episodes: still penalty -29.4 vs gas +10.2.
      Driving normally was punished, swerving (big picture change) wasn't -> off the road.
- [x] 12. Reward v5: still penalty off (gas reward covers standing, off-road rule covers hiding). Resumed from v3 800k.
      1.8 h / 350k steps: **worse** (cubes ~1.0 -> 0.3-0.4, entropy -0.40 -> -1.21, i.e. less and less sure).
      Every resume from v3 got worse: its habits (dawdle, brake) fight each new reward.
- [ ] 13. **v6, from scratch** (Ryan's call: stop building on v3).
      Observation: 2 channels, gray road picture + **cube mask** (any 84x84 cell containing cube pixels = 255):
      far cubes are invisible in the gray picture but a clear dot in the mask.
      Reward: +1 per cube, **cube approach** (phi = sqrt(cube pixels) / 40, reward = phi now - phi before;
      skipped for 3 steps after a pickup because the cube flares up and vanishes over ~2 steps),
      +0.05 gas on the road, -20 losing / off road, no still penalty, no reverse.
      Cube detector (bright orange: r > 150, g > 90, b < 90, r > b + 100): on ~70 screenshots far 160-260 px,
      near 500-1,800, none 0, never anything but cubes. Traced driving straight: approach +0.02..+0.35 per step,
      +1.04 on pickup, and cube 2 comes into view ~6 steps after cube 1 (then lost when driving straight past).
- Recording tool for behavior cloning: `record_demo.py` (saves the same 2-channel obs + your action).
- [x] 14. **Ryan's recordings (6 runs, 5 to the finish) showed the detectors were only right for the start of the
      track**, the only part any agent or test had reached:
      - **The finish:** "YOU WIN / PRESS BACKSPACE TO RESPAWN" on a bright green screen, ~20-25 s (125-160 steps)
        after the start, 10-13 cubes. The game-over check counted it as LOSING (-20). Now: white text + green
        (78-95% of the screen; normal driving at most 21%) = won: WIN_REWARD 30 + 0.1 per step under 300.
      - **On-road:** later sections have darker, bluer lines; the old rule found none there, so every episode
        would have ended there as "off road". New rule (bluish line pixels, threshold 800): on the road median
        ~5,200, 5th pct 1,340, never below 800 for 10 steps in a row; off the road at most 467.
      - **Cubes:** mostly real (very close cubes get huge on screen); fragments after a pickup are cube-coloured
        too, so the pause after a pickup is 6 steps. Live cube count not verified yet (recorded frames are 3 steps apart).
      - Replaying the recorded runs through the new logic: all 5 finished runs end as WON, none cut short.
      - The recorder never ends runs by itself now, catches quick Backspace taps, and saves full-size frames.
- Next: record Ryan driving and learn from it (behavior cloning), then improve with PPO.
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

## Behavior cloning (train_bc.py)

- bc_v1 (all recordings so far: 10 runs, ~1,000 steps, 5 to the finish): learned "always gas" and drove straight
  off at the first curve. Ryan steers with taps (gas, gas+right, gas, ...), so even in curves most steps say gas.
- bc_v2: steering relabelled by intent (the direction tapped most within ±2 steps) and rare actions weighted up.
  Now steers around the first curve, but goes off the road after cube 1 (36-65 steps). On a held-out run it
  predicts Ryan's key only ~55% of the time (memorises the training runs within ~15 epochs): too little data.
- Next: 10-15 more recorded runs, retrain, then PPO fine-tuning from the BC model (RESUME in train.py).

## Watching the agent (viewer.html)

- Training: game 1 of the 10 writes runs/<RUN>_replay.jsonl (a result line for each episode, pictures for
  every 5th). Tests: test.py writes runs/test_<model>.jsonl. Open viewer.html and choose the file.

## PPO from the copy (v7, v8, v7b)

- v7 (from bc_v4, learning rate 1e-4): ~1 cube and longer episodes by 100k (past the curve where the copy failed),
  then learned to stand still (cubes 0 at 300k, episodes to the 1000-step limit), then recovered by itself:
  1.2-1.7 cubes at 475-530k, still rising, but crawling (gas on only 12-25% of steps, ~700 steps per episode).
- v8 (from v7_final + "stalled" rule: no new cube for 90 steps = -20): **failed.** Cubes 0.67 -> 0.1-0.25, ~85%
  stalled, 0-1% gas. Discounting: the agent expects driving to crash (-20 at ~step 40); stalling gives -20 only at
  step 90, which is worth ~40% less now. Putting off the failure beat risking it. Rule switched off (CUBE_TIMEOUT None).
- v7b: continue v7 from v7_final with v7's exact settings, long run.

## Speed: virtual time (tested 2026-10-01, night)

- Real time is the bottleneck: ~6 steps/s per game no matter how fast the PC is (10 games in training: ~39 steps/s).
- **Playwright's page.clock does NOT work with Unity:** after pausing it, the car never moved again (even after
  resuming), though the screen kept rendering.
- **Chrome DevTools virtual time works** (`Emulation.setVirtualTimePolicy`, in core/browser.py `use_virtual_time()`
  / `advance(ms)`): the game is frozen between steps, physics behaves as in real time (driving straight: cube 1, then
  off the road at the first curve), advancing 150 ms of game time takes ~3 ms.
- The screenshot is then the bottleneck: DevTools `Page.captureScreenshot` of the canvas: identical picture, ~39 ms
  vs ~84 ms with Playwright's locator screenshot. A smaller window doesn't help (Unity keeps 960x600, slower to scale).
- 1 game: 12.8 steps/s with virtual time vs 5.3 real time (measured while a 10-game training ran).
- **Real-time steps weren't a fixed amount of game time:** cube 1 at step 13 (real time, 1 game) vs 17 (virtual,
  150 ms/step), so a real-time step was ~0.19 s (1 game) and ~0.26 s in 10-game training. Virtual time makes every
  step identical. To continue from models trained in real time, use STEP_GAME_MS ~250; 150 for a fresh start.
- With virtual time the games are frozen during PPO updates, so RestartAfterUpdate (batch restarts, ~1/3 of episodes
  cut) isn't needed; train.py leaves it out when VIRTUAL_TIME is on.
- Switch: VIRTUAL_TIME in blue_car_env.py (off by default until decided). speed_test.py measures N games.

- **Multi-game speed (2026-10-01 morning, random mostly-gas actions, 60 s each):** virtual time 6 games 60 steps/s,
  8 games 71, 10 games 76, 13 games 78 (saturated: screenshot capacity); real time 10 games 34. So ~2.2x more steps,
  and with 250 ms per step each step also covers ~as much game time as a real-time training step. 8-10 games is the
  sweet spot (13 only adds RAM use: 4.4 GB free).

## v7b overnight result

- 554k -> 1.75M steps, 6.9 h: cubes 1.9 -> ~2.8 (950k-1.55M), best batch 4.5, best episode 7 cubes, no wins.
  Plateaued from ~950k; last 200k slightly lower (2.1). ep_rew -14 -> -7.
