"""Gymnasium environment for the toy car game. The TODOs are yours to write."""
import gymnasium as gym
import numpy as np

from browser_game import BrowserGame

# ---- settings -----------------------------------------------------------
DIFFICULTY = "easy"      # "easy" (brake never needed) or "hard" (must brake before corners)

# Action number -> keys held during that step.
ACTIONS = [
    [],                          # 0 coast
    ["ArrowUp"],                 # 1 gas
    ["ArrowLeft"],               # 2 left
    ["ArrowRight"],              # 3 right
    ["ArrowUp", "ArrowLeft"],    # 4 gas + left
    ["ArrowUp", "ArrowRight"],   # 5 gas + right
    ["ArrowDown"],               # 6 brake
]


class CarEnv(gym.Env):
    def __init__(self, headless=True, difficulty=DIFFICULTY, record_file=None, random_starts=False):
        super().__init__()
        # record_file="runs/my_run.jsonl" saves the car's path every 10th episode for viewer.html
        # random_starts=True: every reset without a seed starts at a random checkpoint (used for evaluation)
        self.game = BrowserGame(headless=headless, difficulty=difficulty, record_file=record_file)
        self.random_starts = random_starts

        # gap[k] = distance from checkpoint k-1 to checkpoint k, for measuring progress between checkpoints
        cps = self.game.page.evaluate("window.getTrack()")["checkpoints"]
        self.gap = [np.hypot(cps[k]["x"] - cps[k - 1]["x"], cps[k]["y"] - cps[k - 1]["y"]) for k in range(len(cps))]
        self.action_space = gym.spaces.Discrete(len(ACTIONS))

        # TODO: self.observation_space = gym.spaces.Box(...)
        self.observation_space = gym.spaces.Box(low= -1, high= 2, shape=(11,), dtype=np.float32)


        self.prev_state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is None and self.random_starts:
            seed = int(self.np_random.integers(2**31))
        state = self.game.reset(seed)   # seed=None: normal start line; a number: random start checkpoint + heading
        self.prev_state = state

        obs = self._get_obs(state)
        return obs, {}

    def step(self, action):
        state = self.game.step(ACTIONS[action])

        # TODO: fill these in
        obs = self._get_obs(state)
        reward = self._get_reward(self.prev_state, state)

        terminated = False
        truncated = False
        
        if(state["crashed"] == True or state["lap"] == True):
            terminated = True
        elif (state["truncated"]):
            truncated = True
        
        info = {"checkpoints": state["checkpointsPassed"], "crashed": state["crashed"], "lap": state["lap"],
                "speed": state["speed"] / state["maxSpeed"], "seconds": state["frame"] * state["dt"]}

        self.prev_state = state
        return obs, reward, terminated, truncated, info

    def close(self):
        self.game.close()

    def _get_obs(self, state):
        speed = state["speed"] / 360
        angle_sin = np.sin(state["angleToCheckpoint"])
        angle_cos = np.cos(state["angleToCheckpoint"])
        dist = state["distToCheckpoint"] / 100
        rays = [r / state["rayMax"] for r in state["rays"]]

        return np.array([speed, angle_sin, angle_cos, dist, *rays], dtype=np.float32)


    def _progress(self, state):
        # How far around the track the car is, in checkpoints: 3.4 = passed 3, 40% of the way to the 4th.
        # Smooth across checkpoints, so no distance goes unpaid (the v1 bug).
        return state["checkpointsPassed"] + 1 - state["distToCheckpoint"] / self.gap[state["nextCheckpoint"]]

    def _get_reward(self, prev_state, state):
        reward = 0.0

        # v3: progress made this step (potential-based shaping). Adds up to ~16 per lap at any speed,
        # so faster laps are always worth more (fewer -0.03 steps), and it still gives a signal every step.
        reward += self._progress(state) - self._progress(prev_state)

        if state["crashed"]:
            reward -= 20 #crashed

        reward -= .03 #ever iteration for trying to be faster

        return reward