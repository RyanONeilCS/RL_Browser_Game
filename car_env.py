"""Gymnasium environment for the toy car game. The TODOs are yours to write."""
from turtle import speed

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
    def __init__(self, headless=True, difficulty=DIFFICULTY, record_file=None):
        super().__init__()
        # record_file="runs/my_run.jsonl" saves the car's path every 10th episode for viewer.html
        self.game = BrowserGame(headless=headless, difficulty=difficulty, record_file=record_file)
        self.action_space = gym.spaces.Discrete(len(ACTIONS))

        # TODO: self.observation_space = gym.spaces.Box(...)
        self.observation_space = gym.spaces.Box(low= -1, high= 2, shape=(11,), dtype=np.float32)


        self.prev_state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        state = self.game.reset()   # TODO (optional): pass a seed for random start positions
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
        
        info = {"checkpoints": state["checkpointsPassed"], "crashed": state["crashed"], "lap": state["lap"]}

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


    def _get_reward(self, prev_state, state):
        reward = 0.0
        passed = state["checkpointsPassed"] > prev_state["checkpointsPassed"]

        if passed:
            reward += 1 #passed a checkpoint
        else:
            reward += (prev_state["distToCheckpoint"] - state["distToCheckpoint"]) / 100

        if state["crashed"]:
            reward -= 10 #crashed

        reward -= .01 #ever iteration for trying to be faster

        return reward