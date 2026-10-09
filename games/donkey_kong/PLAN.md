# Donkey Kong: the plan

What we decided so far (2026-10-07) and why. Facts and measurements about the game are in [NOTES.md](NOTES.md).
Ryan writes all the RL code (env decisions, reward, training); this is the plan to build from, not code.

## Goal

1. Learn the barrel stage (25 m) first: every game goes through it, and it is probably the hardest to learn.
2. Then the other stages (pies, elevators, rivets) with the same observations, actions and reward, so one agent
   can play all of them.

## Actions (7)

| # | Keys | What it does |
|---|---|---|
| 0 | ArrowLeft | walk left |
| 1 | ArrowRight | walk right |
| 2 | ArrowUp | climb up a ladder |
| 3 | ArrowDown | climb down a ladder |
| 4 | Space | jump in place |
| 5 | ArrowLeft + Space | jump left |
| 6 | ArrowRight + Space | jump right |

- **Up and down climb ladders; Space is the only jump.** Without Up, Mario can't climb at all.
- **No "do nothing" action.** Blue Car learned to stall. Waiting is still possible when it's really needed (letting a
  barrel pass before climbing): jump in place, or step left and right.
- Removing "do nothing" doesn't stop stalling by itself (it can still step back and forth). The **reward** has to make
  stalling a bad deal: in Blue Car standing still cost less than crashing, and that is why it stalled.
- Each action must be held long enough for the game to notice it (a 1 ms tap is ignored, 0.15 s works, see NOTES).
- To check by playing: does Down do anything when not on a ladder? Can Mario jump while on a ladder?

## Episodes

- **One episode = one life.** Losing a life costs ~3 s (the "25 m" screen) and the level restarts at the bottom;
  a game over costs much more (Retry = a new game with the ~14 s intro). The checkpoints also start over every life.
- `terminated` = Mario died or reached Pauline. `truncated` = a step limit (time limit only).
- Reset after a death: wait through the "25 m" screen until play starts again. Reset after the last life: Enter on
  Retry (selected by default), then wait out the intro (~14 s).

## Reward v1

| Part | Value (to tune) | Why |
|---|---|---|
| **Height checkpoints**: horizontal lines every **20-30 px** across the whole screen, each paid **once per life** when Mario's feet first get above it | +1 each (example) | Progress up the screen, dense enough to learn from (a few lines per ladder climb), can't be farmed |
| **Death** | about −10 or more, well above one checkpoint | Rushing up a ladder into a barrel must be a bad bet |
| **Reaching Pauline** | big (more than a death), plus a bit for the **BONUS timer left** | Finishing must beat everything else; the bonus left rewards finishing fast |
| Points (score) | **not yet** | Barrel jumps and hammer smashes pay points without climbing: it could learn to farm points at the bottom. Add later (small) once it reaches Pauline. |
| Staying alive | **none** | A reward per step alive teaches hiding / stalling |

Why evenly spaced lines instead of lines placed on each girder:

- **Same on every level**: "higher is better" with no tuning per stage (all four stages are different).
- **Once per life** means nothing can be farmed: climbing down and up again, or jumping, never pays twice.
- **Jumps**: a jump lifts Mario ~27 px (NOTES), so a jump can grab the next line early. Harmless: the total for a
  climb stays the same (that line won't pay again when he climbs), only the timing changes. If the agent starts
  jumping pointlessly at every new height, make the spacing bigger than a jump (30+ px).
- **Partial climbing pays**: about 2-3 lines per gap between girders (girders are ~60 px apart), so it gets paid
  partway up a ladder and discovers that climbing pays. Only paying at the top would be the Blue Car "cube 2" problem
  (reachable, but too rare to find by chance).
- **Won't it rush up ladders into barrels?** Not if death costs enough. Waiting for a barrel to pass costs almost
  nothing (the checkpoints are still there a second later), while rushing risks the death penalty **and** losing
  every checkpoint it would have reached afterwards (the episode ends).
- **Sloped girders and broken ladders** don't matter much with lines every 20-30 px: a broken ladder can pay a line
  once, which is harmless.

Possible extra, only if needed: if the random agent almost never reaches a ladder, add "reached the bottom of a working
ladder" (a fixed list of ladder positions per stage, once per platform). Not planned for now: it's per level.

## Detectors to build (from screenshots) and check

1. **Mario's position**: his pixels by colour (red cap and shirt, blue overalls, skin), the lowest one = his feet.
   Ignore the top of the screen (the lives counter is small Marios). Check that Pauline, Kong, barrels, the oil drum
   (blue!) and the hammer never count. Mario is ~26 px tall.
2. **Death**: the "25 m HOW HIGH CAN YOU GET?" screen (mostly black, Kong + text), or Mario's death animation.
3. **Game over**: the "Game Over" screen.
4. **Reached Pauline / level complete**: not seen yet. Record what happens when you finish the stage.
5. **BONUS timer** (optional): read the digits in the BONUS box (top right). Simpler first version: count steps
   since the life started.

## Checks before the first training run

- [ ] `check_env` passes.
- [ ] Mario detector: print his height every step for a few random episodes, compare with the screenshots
      (whole stage, not just the start: Blue Car's detectors broke three times in parts nothing had reached).
- [ ] **Random-agent test**: how often does it reach the first girder / first checkpoints? If almost never, the
      checkpoints are too sparse (see "Possible extra" above).
- [ ] Break one episode's reward into its parts (checkpoints, death, Pauline) and check nothing unexpected dominates.
- [ ] Compare the *discounted* value of: climbing, stalling at the bottom, and dying early (README: reward design notes).

## Later

- **Points**, small compared with climbing, once it reaches Pauline.
- **Speed**: ~7 steps/s per game in real time, so train 8-10 games at once like Blue Car. Retry frame control
  (virtual time) with the ads blocked: it crashed the page so far, but gave Blue Car ~2x speed and exact steps.
- **Behavior cloning**: record yourself (like Blue Car's `record_demo.py`), copy first, then PPO. The only thing that
  got Blue Car to the finish. Also the way to teach later stages before the agent can reach them.

## Multiple levels

1. Master the barrel stage first.
2. Same observations (crop, size, frame stack), same actions and same reward on every stage; nothing stage-specific.
3. Height checkpoints work on barrels, pies and elevators ("climb to Pauline"). **Rivets** is different (remove 8
   rivets), so it will probably need its own progress signal.
4. A way to start on later stages: test whether **Continue Game** after a game over starts at the stage you reached
   (NOTES). Without it, the agent only sees stage 2 after beating stage 1.
5. Record yourself on every stage (demos for the stages it can't reach yet).
6. Mix stages to avoid forgetting (README: Notes for future games). Save a model per stage.
