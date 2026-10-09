# Donkey Kong: notes

Everything found out about the game so far, and where it is on the
[new-game checklist](../../README.md#adding-a-new-game). RL code (env decisions, reward, training) is Ryan's to write.
**The plan (actions, episodes, reward) is in [PLAN.md](PLAN.md).**

## The game

- Page: https://freekong.org/ (title "Donkey Kong - Play Game Instantly!"). The game runs on the page itself, no iframe.
- Engine: a **Flash game** (`donkey-kong.swf`, ActionScript 2, 530x490 at 30 fps) played by **Ruffle**, a Flash
  emulator in JavaScript/WebAssembly. Ruffle draws inside its own `<ruffle-player>` element, so screenshots use the
  selector `#game-container`, not `canvas` (there is no `<canvas>` in the page itself).
- A remake of the arcade game, not the NES port. The .swf has code for **all four arcade stages**: barrels (25 m),
  pies on conveyor belts (50 m), elevators (75 m) and rivets (100 m). Only 25 m has been seen so far.
- The page has Google ads. With `VIEWPORT = (530, 490)` the game fills the whole window and no ad covered it.

## Found so far (tested from Python, 2026-10-07)

| What | Finding |
|---|---|
| Loads headless? | Yes. ~8 s until the menu shows. |
| Page overlays | A "Click on the game area and press ENTER" bar, a menu button and a full-screen button sit on top of the game (they cover the score). `prepare_page()` hides them. |
| Focus | Click the game once before keys work (`prepare_page()` clicks near the bottom edge). |
| **Key presses** | **A quick tap is missed**: Playwright's `press()` (down + up within ~1 ms) did nothing in the menu. Holding a key ~0.15 s works (`tap()`). The game probably checks keys once per frame (30 fps = 33 ms). Actions must be held for at least a frame or two. |
| Menu | New Game / Continue Game / Controls. Arrow keys move the cursor, Enter selects. Mouse clicks on the menu text do nothing. |
| Start | Enter on New Game -> intro (Kong climbs, bends the girders) -> "25 m HOW HIGH CAN YOU GET?" -> play. **Play starts ~14 s after Enter.** |
| Controls | Arrow keys move left/right (Mario walks ~40 px per second at this size), up/down climb ladders, **Space jumps** (also while walking: a jump to the side). `z` and `x` did nothing visible. |
| HUD | Top: score (left), HIGH SCORE (middle), level `L=01` (right). Below: lives (small Marios, 3 at the start), hammer count, a second item count, and the **BONUS** timer box (top right, starts at 5000 and counts down 100 at a time). |
| Losing a life | Standing at the start, Mario died after ~5 s (the first barrel / oil-drum fire). Then the "25 m" screen for ~3 s and the level starts again with one life fewer and BONUS back to 5000. No long intro. |
| Game over | After the last life: a "Game Over" screen (final score, high score) with Retry / Main Menu / High Scores, Retry selected. Standing still: game over ~16 s after play started. |
| Retry | Enter on Retry = a **brand-new game**: the full ~14 s intro, 3 lives, level 1. |
| Continue Game | Not tested yet (needs a level beaten first). The .swf has no saved data (no SharedObject), so it can only work within one browser session: maybe it continues at the level you reached after a game over. **Worth testing by hand:** beat 25 m, lose all lives, Main Menu -> Continue Game. If it starts at 50 m, that's a way to train on later levels. |
| Screenshot speed | ~67 ms for the 530x490 game -> at most ~15 steps/s for one game, plus the time each action is held. |
| Escape / P | **Escape quits to the main menu.** P does not pause. |
| Frame control (virtual time) | **Doesn't work yet**: with `frame_control=True` the page crashed while loading (probably the ad scripts or Ruffle under Chrome's deterministic mode). Real time works. Worth retrying later with the ads blocked (Playwright `page.route`), since it gave Blue Car ~2x speed and exact steps. |

Screenshots of all of this: run `explore.py` (saved in `screenshots/`, not in git).

## Measurements (barrel stage, 530x490 screenshots, 2026-10-07)

Pixel rows count from the top (row 0 = top edge). Measured on a few frames, so treat them as approximate.

| What | Value |
|---|---|
| Mario's size | ~26 px tall (standing at the start: rows ~429-455) |
| Mario's start | bottom girder, just right of the oil drum (x ~177-194) |
| Walking speed | ~40 px per second |
| Jump | lifts Mario ~27 px (his top: row 429 -> 402 at the peak), in the air ~0.7 s |
| Girders (top edge) | at x = 265: rows 456, 395, 338, 276, 218, 161, 111 (bottom to top), ~57-62 px apart |
| Girder slope | ~10-15 px from one end of a girder to the other (direction alternates) |
| Oil drum | blue, left of Mario's start (x ~86-112): same blue as his overalls, keep it out of the Mario detector |

So a jump lifts Mario about one body height, under half the gap between girders.

## Checklist progress

- [x] 1. Investigate: engine, URL, controls, menus, start, death, game over, restart
- [x] 2. Folder created (`donkey_kong_env.py` plumbing + TODOs, `explore.py`, `train.py` / `play.py` / `test.py` stubs)
- [x] 3. Running from Python: loads headless, overlays hidden, focus, keys (held, not tapped), screenshots
- [x] 4. Timing: real time, ~67 ms per screenshot. Frame control crashed (see above).
- [x] 5. Actions: 7, no "do nothing" (PLAN.md)
- [x] 6. Observations: whole game, grayscale, 84x84, 4 stacked frames (train.py)
- [x] 7. Reward v1: lines every 25 px once per life (+1), death −10, Pauline +30 + up to +10 for speed
- [x] 8. Done and reset: one episode = one life, reset never waits (see "v1" below)
- [x] 9. check_env passes; random agent: best line ~1 (a jump on the bottom girder), dies after 70-230 steps
- [x] 10. Train: v1 -> v1b -> v1c -> v2 -> v2b -> v2c (see "Where it stands" at the end)
- [ ] 11. Evaluate: best so far reaches the 3rd girder in ~85-90% of lives; nothing has finished the stage yet

## v1 (written by Claude on 2026-10-07 night, to run overnight; Ryan may rewrite it)

Built from PLAN.md. Detectors, all measured on screenshots first:

| Detector | How | Measured |
|---|---|---|
| Playing | white pixels of the BONUS box (top right) | ~790-800 while playing, 0 in the intro, "25 m" screen, Game Over |
| Game Over | not playing + white text in rows 120-220 | ~3,800 vs < 720 |
| Main menu | orange logo pixels in rows 10-100 | ~11,700 vs < 130 |
| Mario | blobs with both his bright red and dark navy, then **tracked** | found in every play frame tested; climbing tracked (first ladder: lines 1-2 at the right moments) |
| Died | the BONUS box disappears while Mario was alive, and he was last seen below row 100 | the box goes ~1.5 s after the hit (death animation), so the penalty comes ~10 steps late |
| Won | the box disappears and Mario was last seen above row 100 (Pauline's platform) | **not checked**: nothing has reached the top yet |

What went wrong on the way (worth knowing for later games):

- **Counting girders as "playing"** also fired during the intro, where Kong carries Pauline up the ladders, and **she
  passed as Mario**: the random agent "climbed" to line 14 in 7 s. Fixed by using the BONUS box (only shown in play).
- **Barrels (blue and orange) also pass the colour test.** Fixed by tracking: a life only starts from a Mario seen on
  the bottom girder, and after that only blobs within 40 px (+15 per step he was missed, at most 100) of his last
  position count. Without it, Mario "jumped" 150 px up in 5 steps.
- **Escape quits to the main menu** (the env never presses it; if it ever ends up there, it presses Enter = New Game).
- **P doesn't pause.** So the games keep running during PPO updates; train.py releases the keys meanwhile.
- **Blocking the ads (Playwright `page.route`) broke the keyboard**, even with nothing actually blocked (any request
  interception). Left the ads on; they never covered the game.
- **The game browsers leak memory.** First night: the 10 browsers grew ~1.6 GB an hour and available RAM fell
  ~2.4 GB an hour (after a ~1.6 GB drop in the first half hour while everything started). Fixed after the run had
  started: every 30 min each game's browser is closed and opened again at the next reset after a death
  (`RESTART_BROWSER_MINUTES`). Tested with a 30 s timer: 3 restarts, play continued normally.
  `ram_watchdog.ps1` logs memory to `runs/v1_ram.log` and stops the training if memory gets too low.
- **Lockstep:** the 10 games step together, so reset() never waits for the next life. The waiting screens ("25 m",
  Game Over, the ~14 s intro after Retry) are ordinary steps with reward 0 and no keys. ~40% of steps are waiting.

Training v1: `python games/donkey_kong/train.py` (RUN "v1", 10 games, PPO CnnPolicy, n_steps 512, ent_coef 0.01).
Watch episodes: open `viewer.html` and select these together (Ctrl+click): `runs/v1_replay.jsonl` (every 50th episode
of game 1 with pictures, a result line for all its episodes), `runs/v1_records.jsonl` (every time any game reaches a
line no game has reached before, from line 2 up, and the first win) and `runs/v1_best_episode.jsonl` (the episode with
the highest reward so far, replaced when beaten). `runs/v1_best.json` holds what has been reached so far.
Log of a background run: `runs/v1_train.log`. TensorBoard: `rollout/height` = average best line per episode (random ~1; the first girder
is line 2, every girder ~2.4 lines more, the top ~14), `rollout/died`, `rollout/won`.

## Open questions

- Does Continue Game start at the level you reached? (see the table)
- Order of the stages: arcade order is 25 m -> 75 m -> 100 m on the first loop, 50 m added later. Is it the same here?
- What happens when Mario reaches Pauline (a level-complete screen? straight to the next stage?)
- Does the game kill Mario when the BONUS timer reaches 0, like the arcade game? (Probably: in v2/v2b/v2c every life
  ended in death, even the ones that waited 600+ steps in a safe spot; none reached the 1500-step time limit.)
- How far up a broken ladder can Mario climb?
- Does Down do anything when not on a ladder? Can Mario jump while on a ladder?
- How many points per barrel jumped / hammer smash (the score could be read from the top-left digits)?

## v1 result (first night, 2026-10-07 3:11-6:10)

- ~565k steps, then the RAM watchdog stopped it (the browsers' memory leak; fixed since, see above). Newest checkpoint
  `models/v1_checkpoints/v1_550000_steps`.
- Average best line per life 0.94 -> 1.26, steps per life 77 -> 159, reward -9.2 -> -8.9, no wins.
- First times at a new line (saved in `runs/v1_records.jsonl`): line 3 at ~57k steps and line 4 at ~263k steps were
  **real** climbs (2nd and 3rd girder); line 6 at ~512k was **fake**: Mario was on the bottom girder, the tracker had
  hopped onto a fireball (blue flame with red eyes passes the colour test) and "climbed" with it.
- `approx_kl` grew from 0.01 to 0.21 (healthy 0.01-0.02): updates far too big. Entropy -1.94 -> -1.40 (still exploring).

## v1b (resumes v1's 550k checkpoint)

- Stricter Mario blobs (at least 50 navy and 30 red pixels, Mario's size): the fire and fireballs no longer pass
  (39 of 40 busy frames left with only Mario; 90 of 90 play frames still find him).
- While Mario is lost, he can't reappear more than 32 px (+6 per lost step, at most 70) above his last position; after
  10 steps lost he can be picked up anywhere at or below his last height (falls).
- A crashed page ("Target crashed", seen once in a test) restarts that browser and cuts the episode, instead of
  stopping the whole training. Tested by closing the page mid-episode.
- Learning rate 3e-4 -> 1e-4 and `target_kl` 0.02 (an update stops early once it changes the policy this much).
- Not tested: whether Mario holding the hammer passes the stricter test. If not, he is just "lost" (no height reward)
  until the hammer is gone, which can't create fake reward.

## v1b result and v1c (2026-10-07 evening)

- v1b with the fixes and the ads blocked (from 4:35 PM): average best line 0.93 -> ~2.7 by 1.15M steps, then flat at
  ~2.7-3.0 until 1.67M (~3 h). approx_kl stayed 0.01-0.02. Memory: the browsers sawtooth 6.2-7.1 GB with the 30-min
  restarts (blocking the ads saved ~3 GB but did not stop the leak).
- Watching it: it goes straight for the first ladder (jumping a lot: line 1 is in jump reach, and jumping clears the
  barrels and the oil-drum fireball), then wanders on the 2nd girder. The next working ladder is a long walk away and
  walking sideways earns no height: nothing says which way to go.
- v1c: **exploration reward**, +0.1 the first time Mario's standing position enters each 40x30 px cell in a life
  (standing = his lowest point over the last 6 steps, so jumps don't count). Walking the 2nd girder (~10 cells) is
  worth about one height line. Tested: walking pays a cell every ~40 px, climbing pays cells + lines, jumping in place
  pays nothing extra.
- v1c: **speed bonus up to 45** (was 10): every step before reaching Pauline costs 0.03, more than exploring earns
  (~0.02/step), and the full bonus is far more than all the cells together (~10). (Ryan's point.)
- Not changed (one change at a time): the first height line is still in jump reach.

## v1c result and v2 (2026-10-08)

- v1c (exploration cells, from v1b 1.76M) ran 9:44 PM -> 10:30 AM, 1.76M -> 4.56M steps: **no breakthrough**. Height
  2.75 -> ~2.95 within the first ~1.5 h, then flat all night; cells per life 14 -> ~16; no new record after line 6.
  Training healthy all night (approx_kl 0.014-0.024, 61 fps, RAM steady at ~8 GB free). Stopped early (by Claude, Ryan
  away) to start v2. Newest checkpoint `models/v1c_checkpoints/v1c_4560470_steps`.
- Suspected cause: at 84x84 a ladder was a ~2 px grey smudge and broken / working ladders looked the same.
- v2 (from scratch, the observation changed):
  - **Observation:** the play area only (x 60-480, y 50-470) at 140x140 grayscale, 1 px = 3 game px (was the whole
    530x490 screen at 84x84). Rungs and the gaps in broken ladders are visible.
  - **Ladder reward:** +1 the first time in a life Mario stands at the foot of each of the 9 working ladders of the
    barrel stage (list in `LADDERS`; broken stubs left out). Tested: fires once at the first ladder, then the height
    lines pay for the climb. Only right for the barrel stage.
  - Kept: height lines, exploration cells, death -10, Pauline +30 + up to +45 speed. Learning rate 2e-4 (fresh
    network), target_kl 0.02.
- v2 started 10:30 AM in the background (log `runs/v2_train.log`), 62 fps.

- **One recording file per run** (from v2, Ryan's idea): `runs/<run>_episodes.jsonl`. Game 1 writes a result line for
  every life and pictures for every 50th; every game adds the lives worth keeping with pictures and `"why"`:
  "first time at line N", "first win", "best so far" (beats the previous best reward by at least 1). The 10 games take
  turns through a lock file. Saved lives carry the real training step; waiting screens between lives aren't recorded.
- **viewer.html rewritten:** one file (older runs' several files still work), a chart of the average line per life
  (game 1, last 50 lives, with the middle-half band) plus dots only for lives with pictures (blue rings = special),
  a "Watch" list to jump to any saved life, 4x default speed, waits for pictures instead of skipping them.
- v2 restarted 10:42 AM on this format (the first 12 minutes moved to `runs/old_v2_first_try/`).

## v2 so far and recordings (2026-10-08 evening)

- v2: height 0.98 -> ~3.7 by ~800k steps (2:18 PM breakthrough past v1c's ceiling: ladders per life 1.1 -> 2.6), then a
  plateau at ~3.55-3.85 for 5+ hours. Highest: line 8 (4th girder) once, at ~729k steps. Game 1's 700 logged lives:
  ~37% end on the bottom girder, ~31% on the 2nd, ~31% on the 3rd, none on the 4th. The bottleneck: 3rd -> 4th girder.
- Built for behavior cloning (Ryan to record later; v2 left running):
  - `record_demo.py`: you play in a window (arrow keys + Space); saves the agent's 140x140 picture + the matching action
    each step, only while a life is played; steps with no movement key are saved as -1 (the agent has no "do
    nothing"); presses Enter on Game Over for you. Tested with scripted keys: right / up / jump / no key all read right.
  - `train_bc.py`: stacks 4 pictures like training, drops the -1 steps, weights rare actions up, holds out the highest
    life, and fine-tunes v2's newest checkpoint (lr 1e-4) -> `models/bc_v2`. Tested end to end on a scripted recording.
  - Then: train.py with `RESUME = "models/bc_v2"` and a lower learning rate at first.

## v2 result and v2b (2026-10-09)

- v2 finished at 3.0M steps (12:20 AM, `models/v2_final`): plateau at ~3.55-3.85 from ~1M steps on; highest line 8
  (4th girder) once at ~729k. The 3rd -> 4th girder step is the wall.
- v2b (from `models/v2_final`, started 12:21 AM, one change): **reward for walking toward the right ladder**. Each step,
  +1 per 300 px Mario gets closer to the nearest working ladder up from the girder he stands on (minus as much for
  moving away); no reward on steps where the girder or target changes. Tested live: ~+0.03 per step walking toward the
  first ladder, ~-0.03 away, nothing on arriving at a new girder. Settings otherwise unchanged (lr 2e-4, target_kl 0.02).
- v2b result (12:21 AM -> 1:28 PM, 3.0M -> 5.9M steps, stopped by Ryan to record): the walking reward made the agent
  reach the 3rd girder in ~84% of game 1's lives (v2: ~31%), but almost none climb to the 4th; the average height stayed
  in v2's band (3.3-3.9) and v2b's best was line 6. The wall is now exactly the 3rd -> 4th girder ladder. Newest
  checkpoint `models/v2b_checkpoints/v2b_5900320_steps` (stopped, so no v2b_final). train_bc.py now starts from it.
- Recordings (Ryan, 2026-10-09): 14 sessions, 27 lives, 4 reached Pauline (line 13). Behavior cloning on top of v2b made it
  WORSE in tests (12 sampled lives): all steps (bc_v2) average line 3.12, only line-4+ steps (bc_v2b) 3.08, v2b itself
  3.92 -- fine-tuning the whole network on ~1,700 steps disturbs what v2b does well on the lower girders.
- Ryan watched v2b: it often waits in a safe pocket at the end of the 3rd girder, jumping the barrels as they roll off
  (often after the hammer: Mario can't climb while holding it). Not point farming (score isn't in the reward): avoiding
  death. Data: all lives still end in death, but v2b had 29 lives of 600+ steps (v2: 10), all stuck at line 4.
- v2c (from v2b's newest checkpoint, 2026-10-09 afternoon): **step penalty -0.01 per step while alive** (not on waiting
  screens). Risk Ryan raised: it could make it rush and die; death (-10) still costs far more than 100 steps of waiting.

## v2c result (2026-10-09)

- v2c = v2b + step penalty -0.01 per step while alive. Ran 2:10-3:05 PM (10 games), 3:05-3:08 PM (2 games, while Ryan
  played), 4:56-5:27 PM (10 games); 5.90M -> 6.22M steps, then stopped cleanly with the stop file (`models/v2c_final`).
- Short run, but: height back to ~3.9 (max 4.25 for one report), ladders ~2.9 per life; game 1's 98 lives: 88 ended
  with best line 4 (3rd girder), 8 lasted 600+ steps (v2b: 29 in its last 300), average length 345 steps (v2b ~336), so
  **no rushing** and somewhat less waiting. v2c's best: line 5. Still no 3rd -> 4th girder climb.
- The update-size cap (target_kl 0.02) stopped many updates after 1-2 passes ("Early stopping at step 0/1"): with
  learning rate 2e-4 each pass changes the policy a lot. If continued: learning rate 1e-4.
- Stopped by Ryan to record many more runs first (plan: test how much lots of recordings help, possibly from scratch).

## Where it stands (2026-10-09 evening)

All runs (barrel stage only; "line" = height line, 25 px each; 3rd girder ~ line 4-5, 4th ~ 7-8, Pauline ~ 13-14):

| Run | Start | Change | Steps | Result |
|---|---|---|---|---|
| v1 | scratch | height lines, death -10, 84x84 whole screen | 0 -> 0.55M | ~1.3; tracker drifted onto a fireball (fake line 6); updates too big (KL 0.21) |
| v1b | v1 | stricter Mario tracking, crash recovery, lr 1e-4 + target_kl, browser restarts, ads blocked | -> 1.76M | ~2.7-3.0: climbs the 1st ladder, wanders on the 2nd girder |
| v1c | v1b | exploration cells (+0.1 per new 40x30 cell per life), speed bonus up to 45 | -> 4.56M | no change (~2.95) |
| v2 | scratch | 140x140 crop of the play area, ladder-foot bonus (9 working ladders) | 0 -> 3.0M | breakthrough to ~3.7 at ~0.8M; line 8 once; ~31% of lives reach the 3rd girder |
| v2b | v2 | reward for walking toward the right ladder | -> 5.9M | ~84% reach the 3rd girder, then stall in a safe pocket; best line 6 |
| v2c | v2b | step penalty -0.01 | -> 6.22M | ~88% reach the 3rd girder, less stalling, no rushing; best line 5 |

The wall: getting from the 3rd girder to the 4th (working ladders at x 278 and 407). Behavior cloning of Ryan's 27 lives
on top of v2b made it worse (whole network disturbed by ~1,700 steps).

What the env does now (donkey_kong_env.py):
- Actions (7): left, right, up, down, jump, jump left, jump right; no "do nothing".
- Observation: 140x140 grayscale crop of the play area, 4 stacked frames (train.py).
- Reward per life: +1 per new height line, +1 per working-ladder foot reached, +0.1 per new exploration cell,
  +-1 per 300 px toward/away from the nearest working ladder up from Mario's girder, -0.01 per step alive, -10 death,
  +30 + up to +45 for reaching Pauline fast (never happened). Points (score) are NOT in the reward.
- One episode = one life; reset never waits (the waiting screens are steps with reward 0 and no keys).
- Mario tracking: red + navy blob of Mario's size, must start on the bottom girder, can't jump more than ~1 girder up
  while lost (fireballs / barrels used to fool it).
- Browser per game restarted every 30 min (memory leak, probably Ruffle); ads blocked with Chromium's
  `--host-resolver-rules` (sends ~11 ad / tracking hosts to 0.0.0.0; saves ~3 GB with 10 games; `page.route` broke the
  keyboard); a crashed page restarts that browser instead of stopping the run.

Running training (train.py):
- `python games/donkey_kong/train.py` (10 games, ~63 steps/s, ~14 GB RAM) or `python games/donkey_kong/train.py 2`
  (2 games, ~4 GB, ~13 steps/s). Every batch is 5,120 steps whatever the number of games.
- `RESUME = "latest"`: continues the run from its newest checkpoint or models/<RUN>_final, whichever is newer.
- Stop cleanly: Ctrl+C in its terminal, or create `runs/<RUN>.stop` (finishes the batch, saves, exits).
- `ram_watchdog.ps1`: logs memory every 5 min to runs/v1_ram.log and stops the training below 1.5 GB available.
- Recording file: one per run, `runs/<RUN>_episodes.jsonl` (every 50th life of game 1 + firsts at a new line, first
  win, best so far from all games), open it in viewer.html.

Next (Ryan's plan): record a LOT of runs with record_demo.py, then test how much they help, possibly from scratch.
Ideas for using them better than plain cloning on top of a trained model:
- Clone from scratch (BASE = None in train_bc.py) with many recordings, then PPO from the copy with a low learning rate.
- Demonstration loss during PPO (core/demo_ppo.py) with a small weight (~0.02, fading out), so PPO keeps what works.
- Only the 3rd-girder-and-up part (MIN_LINE = 4) if starting from a trained model.
