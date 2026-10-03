# Blue Car: what we tried, what broke, what fixed it

A retrospective of the whole Blue Car effort (2026-09-29 to 2026-10-02), so that coming back to it later starts
from what we know instead of from scratch. `NOTES.md` has the raw day-by-day log and measurements; this file is the
organised version. Numbers are from our own tests and TensorBoard logs.

## Where it stands

- **Best models:** `models/v7e_checkpoints/v7e_3410440_steps` (most consistent: 8.4 cubes on average in tests,
  usually gets close to the finish) and `v7e_3460440_steps` (reached the finish in 1 of 5 test runs).
- They drive the **whole track from pixels** and sometimes win, but **slowly**: ~3–4 min of game time per lap
  (about 1,300 steps of 200 ms) where Ryan needs ~25 s (125–160 steps) and collects 10–13 cubes.
- **The unsolved problem is speed.** Every change aimed at making it faster ended with PPO drifting back to slow,
  cautious driving.
- `play.py` is set to `v7e_3460440_steps`; the env uses v7e's actions and frame-control virtual time (200 ms steps).
- Total so far: ~8 million training steps, ~33,000 episodes, ~49 hours of training across 12 runs.

## The game, as far as we know it

- Unity WebGL, loaded from `https://html-classic.itch.zone/html/13523181/index.html` (the itch page says
  "CURRENTLY NOT WORKING", but it runs; a local copy from the developer would be safer).
- `w` = gas, `a`/`d` steer (only while moving), `s` reverses (loses right at the start), Backspace restarts any time.
- **YOU LOSE** = white text on a dark screen. **YOU WIN** = the same white text on a **bright green** screen
  (~20–25 s into a good run). Off the road doesn't lose right away (YOU LOSE comes after ~3–12 s).
- The number bottom-left counts **cubes**; the glowing orange cubes are pickups. Cube 1 ~2 s in, cube 2 ~5 s in.
- The track gets darker / bluer and differently lit further in. **Every pixel detector must be checked on
  the whole track, not just the start** (see "Detectors" below: this bit us three times).

## Timeline of training runs

| Run | What changed | What happened | Why / lesson |
|---|---|---|---|
| v1 | +1 cube, −10 lose, −0.01/step; 8 games, real time | Learned to **stand still** (cubes ≈ 0, episodes to the limit) | Standing never loses: −0.01/step is far cheaper than −10. Also slowed overnight from a **memory leak** (Python 3.14 + Playwright, see Infrastructure) |
| v2 | + penalty when the picture barely changes ("still") | Learned to **hide**: drove off the road into a dark spot and wiggled forever | In a near-black picture `equalizeHist` turns grain into "motion", so the still check missed it; off-road had no consequence |
| v3 | Off the road for 10 steps = lost | **Learned to drive** and reach cube 1, but never cube 2; slow, braking a lot | Also fixed: only 18 steps/s (1 s respawn wait × lockstep games), and the on-road threshold punished dim road sections |
| v3b | More exploration (ent_coef 0.01), longer episodes | Worse: exploration mostly meant leaving the road | Cube 2 in 1 of 92 batches: reachable but far too rare to learn from |
| v4 | +0.05 per step with gas on the road, −20 lose, no reverse | Worse: off the road 98–100% | The still check **misfired while driving** (39% of gas steps counted as "still"): normal driving was punished, swerving wasn't |
| v5 | Still penalty off | Worse again (cubes ~1 → 0.3) | Every resume from v3 got worse: its habits fought each new reward |
| v6 | Fresh start: **cube channel** in the observation + **cube approach reward** | (set up, then superseded by v7 before a long run) | Far cubes were invisible in the gray picture; the cube mask makes them clear dots |
| — | Ryan's recordings revealed: the **finish counted as losing** (−20), the **on-road check failed in later sections**, cube fragments miscounted | Fixed all three (see Detectors) | Nothing had ever reached the later track, so none of it had been tested there |
| BC | **Behavior cloning** on Ryan's recordings (`train_bc.py`) | bc_v1 "always gas"; bc_v2–v4 steer, collect cube 1, then turn the wrong way at the next curve | Tap steering, little data, and run-splitting bugs (see Behavior cloning) |
| v7 | PPO from the copy (bc_v4), learning rate 1e-4 | Got past the curve, then **stood still** (200–350k), then **recovered by itself** to 1.2–1.7 cubes | Starting from a copy works; PPO still drifts to "safe" |
| v8 | v7 + **stall rule** (no cube for 90 steps = lost) | Failed: coasted until the timeout (0–1% gas) | **Discounting:** −20 at step 90 is worth ~40% less than −20 at step 40, so delaying failure beat risking it |
| v7b | v7 continued, stall rule off, long run | Steady: 1.9 → ~2.8 cubes, then plateau | |
| v7c | + **frame-control virtual time** + demonstration loss (weight 0.1) | Collapsed 2.8 → 0.8 cubes after one update | The demo loss memorised the 2,500 recorded steps and dragged it back to the plain copy's driving |
| v7d | Virtual time, **no demo loss** | **Best progress:** 3.2 → ~6 cubes, best episode 11 | Most episodes near the end ran out of time (1,000 steps) |
| v7e | Time limit 2,000 steps | **First wins** (up to 18% of a batch, ~1,300-step finishes), then decline to ~2 cubes | Each gas step on the road netted **+0.04**, so longer episodes paid more than finishing (+30) |
| v9 | From v7e 3.41M; step penalty 0.06 (gas on road nets −0.01), speed bonus up to 1,500 steps, lr 5e-5 | Worse: cubes 5.8 → 3, episodes longer | The bigger time penalty did not make it faster |
| v10 | 6 of 7 actions include gas | Failed: 8 → ~0.7 cubes and stuck | Changing the actions threw away too much of what v7e had learned; it crawled with the one coast action |

## Problems and fixes, by area

### Infrastructure / speed

| Problem | Fix |
|---|---|
| Headless Chromium rendered WebGL on the CPU: **2.4 s per screenshot** | GPU flags in `core/browser.py` (`--use-angle=d3d11` etc.) + JPEG: ~67 ms |
| Real time: ~6.7 steps/s per game | Run 8–12 games in parallel (`SubprocVecEnv`); 10 games ≈ 38 steps/s |
| **Memory leak:** each game process grew to ~8 GB overnight (training slowed 45 → 26 steps/s) | Python 3.14's asyncio "awaited by" tracking never empties with Playwright's sync API. **Use Python 3.12** (`.venv` is 3.12) |
| A run accidentally used the old 3.14 venv after renaming it | VS Code followed the renamed folder; select the `.venv` (3.12) interpreter; check `python --version` |
| PPO pauses to update while real-time games keep driving blind | `core/realtime.py` `RestartAfterUpdate` (release keys, restart games each batch); unnecessary with virtual time |
| 1 s respawn wait × games in lockstep: every reset stalled all 10 games (18 steps/s) | Respawn wait 0.2 s (respawn is done in < 0.1 s) |
| Background runs started from Claude were cut after ~30 min | Start long runs as their own process (`Start-Process`, see "How to resume") |
| Playwright `page.clock` to speed the game up | **Doesn't work with Unity:** after pausing, the car never moved again |
| DevTools virtual time alone | Wrong physics: only **1 frame per advance**, so steering barely worked (v7b: 0.8 cubes vs 3.0 real time) |
| **Frame control** (virtual time + `HeadlessExperimental.beginFrame`, one frame per 1/60 s) | Works: exact clock, steering like real time, **62–67 steps/s** with 10 games (~1.7–2× real time). `VIRTUAL_TIME = True` |
| Real-time steps weren't a fixed length (~0.19 s with 1 game, ~0.26 s in 10-game training) | With frame control every step is exactly `STEP_GAME_MS`; **200 ms** matches the real-time-trained models (250 ms made them drive badly) |
| RAM slowly dropping during long runs (~1.3 GB/h) | Windows pages idle memory out on its own; speed was unaffected. Watch free RAM on long runs |

### Detectors (all in `blue_car_env.py`; all measured on screenshots before use)

| Detector | Problem | Fix |
|---|---|---|
| Game over | Counted the **YOU WIN** screen as losing (−20 for finishing!) | `_is_win`: white text + green screen (78–95% green vs ≤ 21% normally) |
| On road | Light-gray lines: fine at the start, but **found no lines in later, darker sections** (would end every episode there); earlier threshold also too high for dim road | Bluish line rule, threshold 800 (on road: median ~5,200; off road: ≤ 467), recalibrated on 283 frames of Ryan's drives |
| Still | Misfired while driving (grain; dark sections) and missed hiding in the dark | Turned off (`STANDING_STILL_PENALTY = 0`) |
| Forward movement (optical flow) | No separation between driving / standing / reversing (grain, look-alike dashes) | Abandoned |
| Cube pickup | The pickup flare and flying fragments double-counted / gave false approach penalties | Ignore the approach reward for `PICKUP_STEPS = 6` after a pickup |
| Counter "0" for splitting recordings | Used the recording's first frame as "0" (it showed 5), so every "5" looked like a restart | Fixed reference `zero_digit.png` from a checked frame |

### Behavior cloning (`train_bc.py`, `record_demo.py`)

| Problem | Fix / result |
|---|---|
| Recorder cut Ryan's runs short (false "lost" in later sections) | Never end runs automatically; only Backspace starts a new run; also saves full-size frames |
| Quick Backspace taps were missed (runs merged) | Count keydown events (`__restarts`) instead of sampling held keys |
| Tap steering (gas, gas+right, gas, …): the copy learned "always gas" | Relabel by intent (dominant steering within ±2 steps) + class weights |
| Too little data: ~55% on an unseen run with 1,000 steps; ~60–75% with 2,500–3,000 | More recordings help most; 10 finished runs ≈ 2,500 steps is still small |
| The copy turned the wrong way at the curve after cube 1 | Mostly data size; PPO from the copy (v7) fixed it |
| Demonstration loss during PPO (weight 0.1) | Harmful (v7c): memorised the recordings, back to the copy's driving. If retried: much smaller (~0.01) |

### Reward (what the agent did with each idea)

- **The agent optimises what you reward, not what you mean.** Every loophole above (standing, hiding, coasting until a
  timeout, crawling for gas reward) was the agent being correct about the reward.
- **Discounting matters:** a penalty later is worth less (0.99^50 ≈ 0.6). Rules that punish "too slow" at a deadline
  make delaying look good.
- **Net per-step reward decides crawl vs hurry:** in v7e, gas on the road netted +0.04/step, so longer was better.
- Current reward (v9 values, kept): +1 per cube, cube approach (potential-based), +0.05 gas on the road,
  −0.06 per step, −20 lost / off road 10 steps, +30 win + 0.02 per step under 1,500.

## What worked / what didn't

**Worked:** behavior cloning as a starting point, then PPO; the off-road rule; the cube channel and cube approach
reward; telling win from lose; frame-control virtual time; a longer time limit; measuring every detector on real
screenshots (and on the whole track); watching episodes (`viewer.html`) instead of only reading numbers.

**Didn't:** rewards built on noisy pixel measurements (still / progress / optical flow); the stall rule; a
demonstration loss at 0.1; a bigger time penalty; gas-only actions; more exploration. PPO repeatedly drifted toward
slow, safe driving after any improvement.

## If we come back: what to try next (most promising first)

1. **More recordings, then a better copy.** Record 10–20 more runs to the finish (steady steering, ~80–90% of top
   speed). Retrain `train_bc.py` (3× the data), then PPO from that copy. A copy that already drives at Ryan's speed
   doesn't need PPO to *discover* speed: the part it never managed.
2. **Mild demo loss with the bigger dataset** (weight ~0.01, decaying), to keep the copied speed while PPO improves.
3. **Lower learning rate with a schedule** or stop at the best checkpoint: several runs peaked and then declined.
4. **Speed as part of the cube reward:** e.g. scale the cube reward by how soon it came after the previous one
   (Ryan: a cube every ~18 steps), rather than a flat time penalty.
5. A **local copy of the game** (ask the developer): removes the itch.io dependency; reading game state from
   the code could give an exact progress value for the reward (the agent would still only see pixels).

## How to resume

- Activate the venv: `.venv\Scripts\Activate.ps1` (`python --version` must say **3.12.13**).
- Watch the best model: `python games/blue_car/play.py` (real time, window).
- Test a model: set `MODEL` in `test.py`, run `python games/blue_car/test.py`; open `viewer.html` with
  `runs/test_<model>.jsonl` to watch the episodes.
- Record: `python games/blue_car/record_demo.py` (Backspace = next run, Ctrl+C = save to `demos/`).
- Copy: `python games/blue_car/train_bc.py` (uses every recording in `demos/`; set `OUT`).
- Train: set `RUN`, `RESUME`, `LEARNING_RATE`, `DEMO_WEIGHT` in `train.py`. For long runs start it as its own process:
  `Start-Process .venv\Scripts\python.exe -ArgumentList "games\blue_car\train.py" -RedirectStandardOutput games\blue_car\runs\<run>_train.log -RedirectStandardError games\blue_car\runs\<run>_train_err.log -WindowStyle Hidden`
  (or simply `python games/blue_car/train.py` in a terminal you keep open).
- Watch progress: `tensorboard --logdir games` and `viewer.html` with `runs/<run>_replay.jsonl`.
- Settings that matter: `VIRTUAL_TIME = True`, `STEP_GAME_MS = 200`, `MAX_STEPS = 2000`, 10 games, actions as in v7e.

**Check before the next run:**
- `train.py` still says `RUN = "v10"`. Give the new run its own name, or it mixes with v10's checkpoints and logs.
- The reward in `blue_car_env.py` still has **v9's values** (`STEP_PENALTY 0.06`, `FAST_BONUS 0.02`,
  `FAST_STEPS 1500`). The best v7e models were trained with `STEP_PENALTY 0.01`, `FAST_BONUS 0.1`,
  `FAST_STEPS 300`. If you continue from v7e, decide which set to use. v9's values did not help.
