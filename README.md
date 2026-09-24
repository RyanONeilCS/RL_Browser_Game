# RL_Browser_Game

## General Ideas for this project 

My goal for this project is to take my friends browser game and use RL to be able to learn and play the game
This is a goal or a first step to create RL algorithms for more complex games and things
Now this is more of a test for more complex RL algorithms and how they interact with a game and the browser
However if I cant get the code to work inside the browser I have my friends permission to use/edit their code.

### Link For the Game

[Blue Car Game](https://kaboochy.itch.io/blue-car)

## Files

```
game/             toy car test game (open game/index.html to drive it yourself)
browser_game.py   runs the game in Chromium so Python can play it (python browser_game.py = random driving)
car_env.py        Gymnasium env: settings (difficulty, actions) at the top, TODOs for obs/reward/done
train.py          train an agent
play.py           watch a trained agent
viewer.html       sped-up replay of training: open it in a browser and pick a recording from runs/
```

One env file per game: new browser games get their own `<game>_env.py` in this folder.

**Watching training:** create the env with `CarEnv(record_file="runs/my_run.jsonl")`. Every 10th episode's
path is saved while training runs. Open `viewer.html` and choose that file (choose it again to see newer episodes).

**Saving weights:** save models to `models/` (e.g. `models/car_easy.zip`) and load them in `play.py`,
so you only train once.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or the CUDA build on the desktop
python -m playwright install chromium
python browser_game.py
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
- [ ] Fill in `car_env.py` (observations, reward, done)
- [ ] Train with SB3 PPO on easy until it laps, then hard
- [ ] Own PPO in PyTorch, then a bare-bones version; compare with SB3
- [ ] Pixel observations (desktop GPU)
- [ ] Blue Car
