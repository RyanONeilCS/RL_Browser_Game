# RL_Browser_Game

## Quick commands

Run from the project folder in a terminal with the venv active (`.venv\Scripts\Activate.ps1`, the prompt then
starts with `(.venv)`). First-time install: see [Setup](#setup). Scripts work from any folder (they find their
own `models/` and `runs/`), so VS Code's Run button works too.

Toy car (`games/toy_car/`):

| Command | What it does |
|---|---|
| `python games/toy_car/train.py` | Train an agent. Settings (track, steps, run names) are in the file. Saves to `games/toy_car/models/` and `runs/`. |
| `python games/toy_car/play.py` | Watch a trained agent drive in a window at real speed. Set `MODEL` and `DIFFICULTY` at the top. |
| `python games/toy_car/test_random_starts.py` | Test a model from 20 random starts (no window): laps finished and lap times. Set `MODEL` and `DIFFICULTY` at the top. |
| `tensorboard --logdir games` | Training graphs for every game (or `--logdir games/toy_car/runs` for one). Open the printed link (`http://localhost:6006/`). See below. |
| open `games/toy_car/viewer.html` | Replay of training in the browser: choose a `games/toy_car/runs/<name>.jsonl` file (choose it again for newer episodes). |
| open `games/toy_car/game/index.html` | Drive the toy car yourself (arrows/WASD, R = reset). Add `?difficulty=hard` for the hard track. |
| `python games/toy_car/browser_game.py` | Plumbing check: random driving, prints steps per second. |
| `python games/blue_car/train.py` | Blue Car: train with 12 games at once (~62 steps/s, 1.5M steps ≈ 7.5 h). Checkpoints every ~50k steps in `games/blue_car/models/v1_checkpoints/`. |
| `python games/blue_car/test.py` | Blue Car: test a model (or a random agent with `MODEL = None`) without a window: how each episode ended, cubes, wins. Saves the episodes to `games/blue_car/runs/test_<model>.jsonl` for the viewer. |
| open `games/blue_car/viewer.html` | Blue Car: watch recorded episodes (training: `runs/<run>_replay.jsonl`, tests: `runs/test_<model>.jsonl`) at up to 64× speed, with a chart of every episode over training. |
| `python games/blue_car/record_demo.py` | Blue Car: record yourself driving (for behavior cloning). Backspace = next run, Ctrl+C = stop and save to `games/blue_car/demos/`. |
| `python games/blue_car/train_bc.py` | Blue Car: behavior cloning: train the network to copy all recordings in `demos/` (a few minutes, GPU). Saves a PPO model that `train.py` can continue from. |
| `python games/blue_car/explore.py` | Blue Car: opens the game, tries each key, saves screenshots to `games/blue_car/screenshots/`, measures steps/s. See `games/blue_car/NOTES.md`. |
| `cd games/toy_car` then `python -c "from stable_baselines3.common.env_checker import check_env; from car_env import CarEnv; env = CarEnv(); check_env(env); env.close(); print('ok')"` | Check the env's format after changing observations/actions. |

**TensorBoard:** start it in a *second* terminal (the first one is busy training; activate the venv there too).
- Left sidebar: tick the runs to compare (e.g. `hard_v3_1` and `hard_v3_run2_1`).
- Filter box at the top: type `lap_time` (also useful: `lap_rate`, `avg_speed`, `eval/mean_reward`).
- Settings (gear icon): smoothing ~0.6–0.8 makes trends easier to see.
- It refreshes every 30 s while training runs (or use the refresh button). Ctrl+C stops it; training is not affected.

## General Ideas for this project 

My goal for this project is to take my friends browser game and use RL to be able to learn and play the game
This is a goal or a first step to create RL algorithms for more complex games and things
Now this is more of a test for more complex RL algorithms and how they interact with a game and the browser
However if I cant get the code to work inside the browser I have my friends permission to use/edit their code.

### Link For the Game

[Blue Car Game](https://kaboochy.itch.io/blue-car)

## Files

```
core/                       general code shared by every game (one copy; games import it, never copy it)
  browser.py                open a page, hold/press keys, screenshots (GPU rendering), run JavaScript
  training_stats.py         avg_speed / lap_rate / lap_time in SB3's output and TensorBoard (racing games)
  realtime.py               for real-time games: restart games around PPO updates, log info values (e.g. cubes)
  replay.py                 records episodes (small frames + per-step data) for a game's replay viewer
games/
  blue_car/                 Blue Car (Unity WebGL, pixels, real time): framework, TODOs for the RL parts
    NOTES.md                what's known about the game + checklist progress (start here)
    blue_car_env.py         env: plumbing done; actions / observations / reward / game over are TODOs
    explore.py              tries keys, saves screenshots, measures speed (no RL)
    record_demo.py          record yourself driving (for behavior cloning)
    train_bc.py             behavior cloning: learn to copy the recordings
    viewer.html             replay of recorded episodes + chart of training progress
    train.py  play.py  test.py
  toy_car/                  the toy car test game
    game/                   the game itself (open game/index.html to drive it yourself)
    browser_game.py         toy-car-specific plumbing: uses the game's JS API (resetGame / stepGame / gameState)
    car_env.py              Gymnasium env: actions, observations, reward, done
    train.py                train an agent (PPO + eval callback that keeps the best model)
    play.py                 watch a trained agent in a window (MODEL / DIFFICULTY at the top)
    test_random_starts.py   lap count + times from 20 random starts, no window (MODEL / DIFFICULTY at the top)
    viewer.html             replay of training (draws the toy car track)
    models/                 saved agents, models/easy/ and models/hard/ (not in git)
    runs/                   TensorBoard logs (one folder per run) and .jsonl replays (not in git)
pyproject.toml              makes core/ importable everywhere (pip install -e . once, see Setup)
```

**Adding a game:** follow the checklist in [Adding a new game](#adding-a-new-game). Each game gets its own
`games/<game>/` folder with its own `models/` and `runs/`, and imports the general parts from `core/`.

**Watching training:** create the env with `CarEnv(record_file="runs/<run name>.jsonl")`. Every 10th episode's
path is saved while training runs. Open `viewer.html` and choose that file (choose it again to see newer episodes).
The viewer shows the current and average speed. Graphs: `tensorboard --logdir games`.

**Saving weights:** save models to `models/easy/` or `models/hard/` and load them in `play.py`,
so you only train once. Give each run one name and use it for the model, the `.jsonl` and `tb_log_name`.

**Models so far** (in `games/toy_car/models/`, not in git; lap time = deterministic, normal start):

| Model | Track | Steps | Reward | Lap |
|---|---|---|---|---|
| `easy/easy_100k` | easy | 100k | progress + checkpoints, −0.03/step, −20 crash | 5.38 s |
| `easy/easy_500k` | easy | 500k | same | 4.80 s |
| `hard/hard_v1_final` | hard | 500k | same | 6.13 s |
| `hard/hard_v1_best_by_reward` | hard | 140k | same | 7.95 s (highest reward, but slow: the progress-reward bug) |
| `hard/v2_best/best_model` | hard | 130k | checkpoints only, −0.03/step, −20 crash | 6.38 s (9/10 random starts) |
| `hard/hard_v2_final` | hard | 500k | same | 7.18 s (correct but too sparse: learned safe, not fast) |
| `hard/v3_best/best_model` | hard | 490k | smooth progress (potential-based), −0.03/step, −20 crash | 6.08 s (20/20 random starts, avg 6.17 s) |
| `hard/hard_v3_final` | hard | 500k | same | 6.07 s (20/20 random starts, avg 6.15 s), still improving at 500k. **Best hard model.** |
| `hard/v3_run2_best/best_model` | hard | 530k | same (v3, second run) | 6.45 s (20/20, avg 6.58 s) |
| `hard/hard_v3_run2_final` | hard | 750k | same (v3, second run) | 6.38 s (17/20, avg 6.58 s): same code, worse run |

Always compare models with `test_random_starts.py` (deterministic, 20 starts), not with `ep_rew_mean`.

## Adding a new game

A checklist for every new game. Work top to bottom: each step is checked before the next, so when something
breaks you know where. Steps marked *(later)* use parts of `core/` that don't exist yet (pixel env, clock control,
`games/_template/`); until then, copy the matching toy car file and adapt it.

**1. Investigate the game (no code yet)**
- [ ] Play it yourself. Write down: the controls (which keys), what counts as progress, what counts as failure
      (crash, death, game over), how to restart, and roughly how long one attempt takes.
- [ ] Find the URL that runs the game itself. On itch.io the game usually runs inside an embedded frame:
      open the browser dev tools (F12), find the `<iframe>`, and use its `src` URL. Or get a local copy
      (web export) from the developer and put it in `games/<game>/game/`.
- [ ] Note the engine (Godot, Unity, GameMaker, plain JavaScript…) and whether the game draws to a `<canvas>`.
- [ ] Look for anything that blocks automation: "click to start", sound prompts, cookie banners, login, ads,
      menus before play starts. Each one needs a scripted click/key press in reset.
- [ ] If you have the code: check how the game handles time (browser clock / `requestAnimationFrame`?) and
      whether it has state you can read (score, position) to verify your screen readers later.

**2. Create the folder**
- [ ] `games/<game>/` with an env file (`<game>_env.py`), `train.py`, `play.py` and a test script.
      *(later)* copy `games/_template/` instead.
- [ ] Every script starts with the two `os.chdir(...)` lines (see the toy car scripts), so `models/` and `runs/`
      are this game's folders.

**3. Get the game running from Python** (with `core/browser.py`)
- [ ] `Browser(url, headless=False)` opens the game and you can see it.
- [ ] Get past start screens with `press(...)` or clicks, and wait until the game is actually playable.
- [ ] Keys work: `hold([...])` moves the player. If nothing happens, click the canvas first (the game needs focus).
- [ ] `screenshot()` shows the game (set `selector` if the game isn't the first `<canvas>`).
- [ ] It also works with `headless=True` (some WebGL games render differently or not at all headless).

**4. Timing**
- [ ] Measure steps per second. Real time (holding keys and sleeping) is ~15 steps/s: 500k steps ≈ 9 hours.
- [ ] Faster if possible: the game's own step function (like the toy car's `stepGame`), or *(later)* clock
      control that pauses the browser's time and fast-forwards it by exact amounts.
- [ ] Decide how long one action lasts (the toy car uses 4 frames = 1/15 s).

**5. Actions**
- [ ] A short `ACTIONS` list of key combinations, like the toy car's 7. Fewer actions = faster learning.
      Include "do nothing" and only the combinations that make sense (e.g. gas + left).

**6. Observations**
- [ ] State numbers if the game exposes them (fast to learn, but only for games you can read), otherwise pixels.
- [ ] Pixels: crop to the play area, grayscale, shrink (e.g. 84×84), stack the last 4 frames so motion is
      visible, use `CnnPolicy`. *(later)* the pixel env in `core/` does this.
- [ ] Everything relative to the player and on a fixed scale (see Testing notes); no absolute positions.

**7. Reward**
- [ ] Progress: something that goes up as the player does better. Prefer "change in progress" each step
      (potential-based, like the toy car's v3) over one-off bonuses: dense but can't be gamed.
- [ ] Time penalty per step if faster is better; failure penalty well above `time penalty / (1 − γ)`.
- [ ] From the screen if needed: a score number, a progress bar, a colour that means "crashed"…
      If you have the code, compare your screen reader against the true value on a few episodes.
- [ ] Reason about gaming it: compare the *discounted* value of good play, idling, and failing early.

**8. Done and reset**
- [ ] `terminated` = real end (finished, died, crashed). `truncated` = time limit only.
- [ ] Detect game over reliably (from the screen or state), then restart the same way every time.
- [ ] Random starts if the game allows it (for testing whether the agent learned or memorized).

**9. Check the env before training**
- [ ] `check_env` passes.
- [ ] Random agent for a few episodes: episodes end, resets work, rewards look sensible.
- [ ] Break one episode's reward into its parts and check nothing unexpected dominates.
- [ ] Put useful numbers in `info` (progress, time, success) for logging; a replay viewer is optional.

**10. Train**
- [ ] Pick a run name and use it everywhere (`record_file`, `tb_log_name`, best model folder, final model).
- [ ] `EvalCallback` on a *separate* env (random starts, several episodes) to keep the best model.
- [ ] Short run first (~20k steps): does reward go up at all? Then a long run.
- [ ] Watch TensorBoard (`tensorboard --logdir games`) and the replay/`play.py` while it trains.

**11. Evaluate and record**
- [ ] Deterministic test from several starts (like `test_random_starts.py`), compared against a baseline
      (a scripted player, your own score, or the previous model).
- [ ] If results are close, train more than one run: runs vary a lot (see Training notes).
- [ ] Add the model and what you learned to this README, and commit the code (models aren't in git).

## Setup

```bash
py -V:Astral/CPython3.12.13 -m venv .venv    # Python 3.12, not 3.14 (memory leak, see requirements.txt)
.venv\Scripts\activate
pip install -r requirements.txt
pip uninstall -y torch                                              # remove the CPU build SB3 pulled in
pip install torch --index-url https://download.pytorch.org/whl/cu128 # desktop GPU (laptop: .../whl/cpu)
python -m playwright install chromium
pip install -e .                  # makes core/ importable from the game folders (once)
python games/toy_car/browser_game.py
```

## Toy car game

- Controls: Arrows/WASD, R to reset, Shift+D for the debug overlay, H for the HUD.
  Open `game/index.html?difficulty=hard&debug` for hard mode with the overlay.
- `easy`: full throttle is fastest, the brake is never needed. `hard`: you have to brake before corners.
- State fields: `x, y, angle, speed, maxSpeed, nextCheckpoint, numCheckpoints, checkpointsPassed, lap,
  angleToCheckpoint, distToCheckpoint, distFromCenter, halfWidth, rays[7], rayMax, crashed, done,
  truncated, frame, dt, difficulty`.
- Scripted reference lap times (not RL):

  | Driver | easy | hard |
  |---|---|---|
  | Full throttle | 4.9 s | crashes |
  | Best safe constant speed | 4.9 s | 7.7 s |
  | Brakes before corners | 5.8 s | 6.9 s |

## Reward design notes

**The agent optimizes what you reward, not what you meant.** When results look odd, break one lap's
reward into its parts (checkpoints, progress, time, crash) for a fast and a slow model and see which
one "wins". That is how the v1 bug below was found.

**Compare discounted returns, not plain totals.** PPO maximizes Σ γᵗ·rₜ (SB3 default γ = 0.99), so when
checking whether a reward can be gamed (e.g. "is crashing better than idling?"), compare the *discounted*
value of each option.
- The agent effectively looks about `1 / (1 − γ)` steps ahead (γ = 0.99 → ~100 steps).
- A constant per-step penalty `p` can never add up to more than `p / (1 − γ)`
  (−0.01 per step at γ = 0.99 → at most −1, no matter how long the episode is).
- A reward `k` steps away is worth γᵏ of its value now (+1 fifty steps away at γ = 0.99 → ~0.6).
- Example: toy car idling until the 90 s timeout is −13.5 as a plain sum but only about −1 discounted,
  so a −10 crash penalty is already much worse than idling.

**Dense vs sparse rewards.**
- *Dense* (a signal every step, e.g. "you got closer"): learns fast, but easy to get subtly wrong.
- *Sparse* (only at events, e.g. "+1 per checkpoint"): harder to get wrong, but much harder to learn,
  because the agent must work out which of many steps earned the reward. Small differences
  (a slightly faster lap = a few fewer −0.03 steps) get lost in the noise.
- **Potential-based shaping** gets both: pick Φ(state) = "how far along am I" and reward Φ(new) − Φ(old)
  each step. It adds up to Φ(end) − Φ(start), so it can't be gamed by *how* you get there, and it is
  proven not to change which policy is best (Ng et al., 1999); it only speeds up learning.

**Toy car reward history** (hard track):

| Version | Reward | Result | Lesson |
|---|---|---|---|
| v1 | +1 per checkpoint, else (distance to checkpoint closed)/100; −0.03/step; −20 crash | 6.13 s | The distance driven on checkpoint steps was never paid, and faster cars lose more per checkpoint, so **slower laps got more reward** (a 7.95 s lap scored 22.7 vs 22.1 for 6.13 s). |
| v2 | +1 per checkpoint; −0.03/step; −20 crash | 7.18 s | Correct (lap = 16 − 0.03 × steps) but too sparse: learned to get round safely, not fast. |
| v3 | Φ = checkpoints passed + fraction of the way to the next; reward Φ(new) − Φ(old); −0.03/step; −20 crash | 6.07 s | Smooth progress with no gap at checkpoints: ~16 per lap at any speed, so faster is always better. Slower start than v1 but kept improving to the end; learned to brake (10% of steps) and use more gas. Best so far. |

**Penalty sizes:** once the reward is correct, the size of the time penalty doesn't change which lap is
best (any penalty > 0 makes faster better), only how hard it pushes. Bigger = pushier but more crashes while
learning. Keep the crash penalty well above `time penalty / (1 − γ)` so crashing never looks like an escape.

## Training notes

**Reading SB3's output:**

| Stat | Meaning |
|---|---|
| `ep_rew_mean`, `ep_len_mean` | Average reward and length of recent episodes. Length goes *up* while it learns to survive, then *down* as laps get faster. Can't be compared between runs with different rewards. |
| `avg_speed`, `lap_rate`, `lap_time` | From `training_stats.py`. Use `lap_time` to compare runs. |
| `entropy_loss` | How random the policy is. −ln(7) = −1.95 is fully random over 7 actions; moves toward 0 as it gets sure. Near 0 while still bad = stopped exploring (raise `ent_coef`). |
| `approx_kl`, `clip_fraction` | How much each update changes the policy. KL ~0.01–0.02 is healthy; often > 0.05 = updates too big. |
| `explained_variance` | How well the critic predicts returns (1 = perfect). Noisy early, should rise. |
| `eval/mean_reward` | Deterministic test score from the EvalCallback: the least noisy progress signal. |

- **Steps vs epochs:** `total_timesteps` = how long to train. `n_epochs` (default 10) = how many passes PPO
  makes over each batch of `n_steps` (2048) before collecting new data. More epochs reuse data more but
  risk unstable updates. To train longer, raise `total_timesteps`.
- **Longer isn't always better.** Easy plateaued at ~250k (4.8 s), hard at ~150–250k. After the peak the
  policy keeps wobbling at a constant learning rate and can get *worse* (hard v1: 6.65 s at 250k → 6.88 s at 400k).
  Fixes: keep the best model with `EvalCallback` (done in `train.py`), a decaying learning rate, or stop earlier.
- **EvalCallback** runs deterministic test episodes on a *separate* env every `eval_freq` steps and saves
  `best_model.zip` when the score beats the best **of that run** (it doesn't know about earlier runs). It picks
  by reward, so it's only as good as the reward (it picked the slow v1 model). With one fixed start it can pick
  a model that is best at one lap only; use `random_starts=True` and several episodes.
- **Every `PPO(...)` starts from random weights.** To continue from a saved model:
  `PPO.load(path, env=env)` then `learn(..., reset_num_timesteps=False)`.
- **Runs vary a lot.** Two identical v3 runs: 6.15 s vs 6.58 s average from random starts (run 2 was ahead at
  200k, then settled into a more careful style and never caught up, even with 250k more steps). So one run per
  version can be luck: to prove one reward beats another, train 3–5 runs of each and compare averages.
  `PPO(..., seed=n)` makes a single run repeatable.
- **One change at a time**, so you know what caused a result. Runs are cheap (~700 steps/s, 500k ≈ 12–15 min).
- Run `check_env` (from `stable_baselines3.common.env_checker`) on a new env before training: it catches
  shape, dtype and bounds mistakes.

## Testing notes

- **Sampled vs deterministic.** During training actions are *sampled* (it explores), so the viewer and
  `lap_time` look worse than the agent really is (easy 100k: 5.2–6.4 s sampled vs 5.38 s deterministic).
  Judge a model with `deterministic=True`.
- **Memorized or learned?** Test from random starts (`env.reset(seed=n)` = random checkpoint + heading).
  The easy model lapped 20/20 at the same speed from anywhere: it learned to drive, because every observation
  is relative to the car (speed, angle/distance to next checkpoint, rays), never its position on the map.
- **Fixed observation scaling.** Speed is divided by 360 on both tracks (not each track's own max), so a value
  means the same real speed everywhere and an easy model can be moved to hard.
- **Know the limit before chasing it.** On hard, grip caps speed in curves (v ≤ √(520·R)): 360 is only possible
  on ~27% of the track, and a perfect centerline lap averages ~230 px/s and takes ~6.3 s from a standstill.
  "Never reaches top speed" was mostly physics, not the reward.
- Results so far vs the scripted drivers: easy 4.80 s (full throttle: 4.9 s), hard 6.13 s (braking script: 6.9 s).
  The agent beats the scripts by finding a shorter line through the corners.

## Workflow notes

- Each game's scripts start with `os.chdir(<the script's folder>)`, so `models/...` and `runs/...` always mean
  that game's folders, wherever you run the script from.
- Give each run one name and use it everywhere: `record_file="runs/<name>.jsonl"`, `tb_log_name="<name>"`,
  `best_model_save_path="models/<track>/<name>_best/"`, `model.save("models/<track>/<name>_final")`.
  Otherwise the next run silently overwrites the last model.
- `models/` and `runs/` are not in git: trained models only exist on the machine that trained them.
- **Use Python 3.12.** On Python 3.14, Playwright's sync API keeps every call's result in memory forever
  (3.14's new asyncio "awaited by" tracking never empties with Playwright's greenlets). Blue Car's 8 games
  grew to ~8 GB each overnight and training slowed from 45 to 26 steps/s. Found by watching memory per
  process, then `tracemalloc` (showed 3,000 kept screenshots), then following the references.
- For long runs, check memory after the first hour (Task Manager): steady is fine, climbing = a leak.
- Activate the venv in every new terminal (`.venv\Scripts\Activate.ps1`) and run scripts with `python games/toy_car/train.py`.

## Notes for future games

**Games with multiple levels:** not needed for the toy car or Blue Car (one track each), but for later:

- Keep training going across levels (load the previous model and continue) instead of retraining per level.
- Watch out for *catastrophic forgetting*: training only on the newest level makes the agent worse at old ones.
  Fix: keep mixing earlier levels in (e.g. on each reset, pick the newest level ~50% of the time and older ones otherwise).
- Use the same observations and actions on every level: relative to the player, fixed scaling (not per-level
  max values), no absolute positions.
- Move to the next level when the agent is ready (e.g. clears the current one ~80% of the time), not on a timer.
- Save a model at each level (`models/<game>_level1.zip`, ...) so you can go back, and test on all levels
  to catch forgetting early.

**Open-world / long games (e.g. Hollow Knight):** plain PPO on "beat the game" won't work. The real goals
are hours apart, random button mashing never finds the next area, the agent needs memory, it only runs in
real time, and there's no easy reset or access to game data. Main idea: **break the game into small parts
and train on those.** Techniques, roughly from most to least practical:

1. **Split into small, repeatable tasks.** Train on one part at a time, e.g. a single boss fight with
   reward = damage dealt − damage taken, reset by re-entering the fight. People have trained agents on
   Hollow Knight bosses (e.g. Hornet) this way. It's like Blue Car: pixels + key presses + reward from the screen.
2. **Read game data with a mod.** Hollow Knight has a modding API. A small mod could expose HP, position,
   and boss HP to Python and handle resets, giving clean rewards instead of reading them from pixels
   (same idea as the Blue Car "bridge script").
3. **Memory.** The agent must remember things it can't see right now (where it's been, what's unlocked).
   Use a recurrent policy (LSTM) or a transformer; SB3 has `RecurrentPPO` in `sb3-contrib`.
4. **Curiosity / exploration rewards.** Give a bonus for reaching new places or seeing new screens
   (e.g. RND, or counting visited rooms), so the agent explores even when the real reward is far away.
   This is how agents progress in games like Montezuma's Revenge.
5. **Learn from human play first.** Record yourself playing, train the network to copy your actions
   (behavior cloning), then improve it with RL. That skips a huge amount of random exploration.
   OpenAI's Minecraft agent (VPT) did this.
6. **Hierarchy.** A high-level policy picks goals ("go to area X", "fight this boss") and low-level
   policies (e.g. the boss-fight agent from 1) carry them out. Powerful but still an active research area.

**Combining them** (they solve different problems, so real projects stack them). Example pipeline:
1. Mod the game (data, rewards, resets). Everything else relies on this.
2. Split the game into skills: boss fight, platforming a room, travelling A → B.
3. Per skill: record a few human runs → behavior cloning so the agent starts at roughly your level.
4. Improve each skill with PPO + a memory policy (`RecurrentPPO`), using the mod's rewards.
5. Add a curiosity bonus for skills where exploring matters (navigation).
6. Hierarchy on top: start with a **hand-written** "if in area X, run skill Y" controller; replace it
   with a learned high-level policy later.

Cautions: add **one technique at a time** (otherwise you can't tell what broke), keep the curiosity
bonus **small** compared with the real reward, and after behavior cloning use a lower learning rate at
first so PPO doesn't wipe out what it copied from you.

**Recording human play (for behavior cloning):**
- A plain video or screen share isn't enough: the agent needs **(observation, keys pressed)** pairs, synced
  frame by frame. Log the keys alongside the screen/state.
- Record first, train afterwards (not live): a saved dataset can be reused for many training runs and for
  comparing agents, and bad runs can be deleted.
- Save a dataset per session (e.g. `demos/<game>_run01.npz`), not a video: per step, the observation **in the
  same format the env produces** (state numbers or a shrunk 84×84 grayscale frame), the action index from the
  `ACTIONS` list, and optionally reward/done.
- Toy car: the game already knows the keys; a small `record_human.py` (real-time mode) can log state + action.
  Desktop games: screen capture + a keyboard logger (`keyboard` or `pynput`).
- **DAgger** (Dataset Aggregation): after behavior cloning, let the *agent* drive while you correct it
  when it makes mistakes; your corrections are added to the dataset and it retrains. This fixes the main
  weakness of plain cloning: the agent never saw how to recover from situations you never got into.
- More demos isn't always better: a few clean, consistent runs beat many sloppy ones. Include some
  recoveries (e.g. getting back to the middle after drifting wide) so it learns how to correct.
- Pure imitation is capped at roughly your skill level; RL afterwards is what lets it get better than you.
- Advanced: without key logs, a separate model can guess actions from video (OpenAI VPT did this for
  Minecraft YouTube footage). Not needed when recording yourself.

For a non-browser game, the env file uses desktop screen capture (e.g. `mss`) and key presses
(e.g. `pydirectinput`) instead of Playwright. Suggested path: toy car → Blue Car (pixels, real-time,
reward from the screen) → one Hollow Knight boss → add memory / human demos / hierarchy as needed.

**Easy → hard on the toy car** is the same idea on a small scale: load the easy model, keep training on hard,
save as a new file. Expect a dip at first (it drives like on easy and crashes at corners). If it never
learns to brake, raise the exploration bonus (`ent_coef` in SB3).

## Roadmap

- [x] Toy car game + browser plumbing
- [x] Fill in `car_env.py` (observations, reward, done)
- [x] Train with SB3 PPO on easy until it laps (4.80 s), then hard (6.13 s)
- [ ] Hard with the v3 reward; easy model → hard transfer
- [ ] Own PPO in PyTorch, then a bare-bones version; compare with SB3
- [ ] Pixel observations (desktop GPU)
- [ ] Blue Car
