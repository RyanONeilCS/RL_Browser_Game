# RL_Browser_Game

## General Ideas for this project 

My goal for this project is to take my friends browser game and use RL to be able to learn and play the game
This is a goal or a first step to create RL algorithms for more complex games and things
Now this is more of a test for more complex RL algorithms and how they interact with a game and the browser
However if I cant get the code to work inside the browser I have my friends permission to use/edit their code.

### Link For the Game

[Blue Car Game](https://kaboochy.itch.io/blue-car)

## Layout

```
rl_games/core/adapter.py    Adapter interface: start/reset/step/get_state/get_frame/close
rl_games/core/config.py     loads configs/*.yaml
rl_games/core/env_base.py   GameEnv (Gymnasium) SKELETON: observations/reward/termination are TODO
rl_games/adapters/browser.py  Playwright adapter (keys, canvas screenshots, JS hooks, iframes)
games/toy_car/              test game: open index.html in a browser to drive it yourself
configs/toy_car.yaml        actions + JS hooks for the toy car
configs/blue_car.yaml       draft config for Blue Car (phase 5)
scripts/smoke_test.py       random actions through the adapter, no learning
```

The RL code only sees `GameEnv`. Each game is a config (URL, actions as key lists, hooks) plus an adapter,
so adding a game doesn't change the learning code.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or the CUDA build on the desktop
python -m playwright install chromium
python scripts/smoke_test.py           # add --headless to hide the browser
```

## Toy car game

- Play it yourself: open `games/toy_car/index.html`. Arrows/WASD to drive, R to reset, Shift+D for the debug overlay, H for the HUD.
- Stepped mode: `stepGame(keys, frames)` advances exact frames (deterministic, ~170 steps/s headless).
  Real-time mode: the game runs on its own clock and Python holds keys (~12 steps/s).
- State fields (`window.gameState`): `x, y, angle, speed, maxSpeed, nextCheckpoint, numCheckpoints,
  checkpointsPassed, lap, angleToCheckpoint, distToCheckpoint, distFromCenter, halfWidth, rays[7], rayMax,
  crashed, done, truncated, frame, dt, difficulty`.
- Checkpoints are gate lines across the road; a crash on the same frame as crossing one counts as a crash.
- Difficulty: set `window.gameConfig.difficulty` in `configs/toy_car.yaml` (or open `index.html?difficulty=hard`).
  - `easy`: steering works at any speed, so full throttle is optimal and the brake is never needed.
  - `hard`: grip limits turning at speed, coasting barely slows the car, the road is narrower with tighter
    corners. You have to brake before corners.
- Scripted reference lap times (not RL):

  | Driver | easy | hard |
  |---|---|---|
  | Full throttle | 4.9 s | crashes |
  | Best safe constant speed | 4.9 s | 7.7 s (speed 160) |
  | Brakes before corners | 5.8 s | 6.9 s |

## Roadmap

- [x] Scaffold, toy car game, browser adapter, env skeleton
- [ ] Fill in `GameEnv` (observations, reward, termination) and pass `check_env`
- [ ] Train with SB3 PPO on state observations until it laps
- [ ] Own PPO in PyTorch, then a bare-bones version; compare all three with SB3
- [ ] Parallel envs, action repeat, pixel observations (desktop GPU)
- [ ] Blue Car: pixels + screen-based rewards, real-time mode
- [ ] Desktop adapter for non-browser games
