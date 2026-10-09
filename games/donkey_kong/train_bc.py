"""Behavior cloning: train the agent to copy your recorded play (demos/*.npz from record_demo.py).

    python games/donkey_kong/train_bc.py

Result: models/bc_v2.zip, a normal PPO model with the same network as train.py, so it can be tested with test.py /
play.py and improved further with PPO (train.py: RESUME = "models/bc_v2", with a lower learning rate at first so PPO
doesn't wipe out what it copied, see the README).

What it does:
1. Loads every recording and splits it into lives (episode_starts).
2. Stacks the last 4 pictures exactly like VecFrameStack in training (zeros before a life's first step).
3. Leaves out the steps where you pressed no movement key (action -1): the agent has no "do nothing" action.
4. Starts from BASE (by default v2b's newest checkpoint, so it keeps what PPO already learned) and trains the policy to
   predict your action from the stacked pictures (cross-entropy, rarer actions weighted up so "always walk right"
   isn't the easy answer). One life is held out to check that it generalises and to pick the number of epochs.

v1 of this file written by Claude on 2026-10-08 (adapted from games/blue_car/train_bc.py).
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))   # demos/ and models/ are relative to this game's folder

import glob

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

import donkey_kong_env as m

OUT = "models/bc_v2b"   # bc_v2 = first try (all steps)
BASE = "latest:v2b"    # start from this model ("latest:<run>" = its newest checkpoint, a path, or None = fresh network).
                       # v2b: reaches the 3rd girder in ~84% of lives (v2: ~31%), so the recordings only need to teach the rest.
N_STACK = 4            # must match train.py (VecFrameStack n_stack)
EPOCHS = 30
BATCH = 64
LR = 1e-4              # small: fine-tunes the PPO model instead of overwriting it
MIN_LINE = 4           # only copy steps once that life has reached this height line (4 = the 3rd girder). v2b already
                       # reaches the 3rd girder in ~84% of lives; copying ALL of Ryan's play (bc_v2, first try) made it
                       # worse there (8 test lives: average line 3.12 vs v2b 3.88). 0 = copy everything.


# ---- 1. load the recordings, split into lives --------------------------------------------------------

def load_lives():
    """List of (obs, actions, heights, best_line) per recorded life (heights = best line so far, per step)."""
    lives = []
    for path in sorted(glob.glob("demos/donkey_kong_*.npz")):
        d = np.load(path)
        obs, act, starts = d["obs"], d["actions"], d["episode_starts"]
        heights = d["heights"] if "heights" in d else np.zeros(len(act), int)
        bounds = list(np.nonzero(starts)[0]) + [len(act)]
        if not bounds or bounds[0] != 0:
            bounds = [0] + bounds
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b - a >= 10:
                lives.append((obs[a:b], act[a:b], heights[a:b], int(heights[a:b].max())))
        print(f"{os.path.basename(path)}: {len(act)} steps, {len(bounds) - 1} lives")
    return lives


# ---- 2. stack frames like VecFrameStack, then drop the no-key steps -------------------------------------

def stack(obs):
    """(T, H, W, C) -> (T, H, W, N_STACK*C): oldest first, newest last, zeros before the life's start."""
    T, H, W, C = obs.shape
    padded = np.concatenate([np.zeros((N_STACK - 1, H, W, C), obs.dtype), obs])
    return np.concatenate([padded[k:k + T] for k in range(N_STACK)], axis=3)


def life_data(obs, act, heights):
    X = stack(obs)                             # stack first, so the picture history before kept steps is complete
    keep = (act >= 0) & (heights >= MIN_LINE)  # no movement key = not an action the agent has; MIN_LINE: see above
    return X[keep], act[keep]


# ---- 3. the model ------------------------------------------------------------------------------------

class SpacesOnly(gym.Env):
    """Just DonkeyKongEnv's observation/action spaces, so PPO builds exactly the network train.py uses."""
    observation_space = gym.spaces.Box(0, 255, (m.OBS_SIZE, m.OBS_SIZE, 1), np.uint8)
    action_space = gym.spaces.Discrete(len(m.ACTIONS))

    def reset(self, seed=None, options=None):
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, False, {}


def base_model(device):
    env = VecFrameStack(DummyVecEnv([SpacesOnly]), n_stack=N_STACK)
    path = BASE
    if path and path.startswith("latest:"):
        run = path.split(":")[1]
        files = glob.glob(f"models/{run}_checkpoints/{run}_*_steps.zip")
        path = max(files, key=lambda f: int(f.rsplit("_", 2)[-2]))[:-4] if files else None
    if path:
        print(f"starting from {path}")
        return PPO.load(path, env=env, device=device)
    print("starting from a fresh network")
    return PPO("CnnPolicy", env, verbose=0, device=device)


def to_tensor(x, device):
    return torch.as_tensor(x.transpose(0, 3, 1, 2), device=device)   # channels first, like VecTransposeImage


def evaluate(policy, X, y, device):
    policy.set_training_mode(False)
    with torch.no_grad():
        loss, correct = 0.0, 0
        for i in range(0, len(X), 256):
            obs = to_tensor(X[i:i + 256], device)
            acts = torch.as_tensor(y[i:i + 256], device=device)
            _, log_prob, _ = policy.evaluate_actions(obs, acts)
            loss -= log_prob.sum().item()
            correct += (policy.get_distribution(obs).distribution.probs.argmax(1) == acts).sum().item()
    return loss / len(X), correct / len(X)


def class_weights(y):
    """Rarer actions count more (inverse frequency, square-rooted)."""
    counts = np.bincount(y, minlength=len(m.ACTIONS)).astype(float)
    w = np.where(counts > 0, (counts.sum() / np.maximum(counts, 1)) ** 0.5, 0.0)
    return w / w[counts > 0].mean()


def train(X, y, epochs, device, X_val=None, y_val=None):
    model = base_model(device)
    policy = model.policy
    opt = torch.optim.Adam(policy.parameters(), lr=LR)
    rng = np.random.default_rng(0)
    weights = torch.as_tensor(class_weights(y), dtype=torch.float32, device=device)
    best = (float("inf"), 0)
    if X_val is not None:
        va_loss, va_acc = evaluate(policy, X_val, y_val, device)
        print(f"before: held-out life loss {va_loss:.3f} acc {va_acc:.0%} (how close the base model already is)")
    for epoch in range(1, epochs + 1):
        policy.set_training_mode(True)
        order = rng.permutation(len(X))
        for i in range(0, len(X), BATCH):
            idx = order[i:i + BATCH]
            acts = torch.as_tensor(y[idx], device=device)
            _, log_prob, _ = policy.evaluate_actions(to_tensor(X[idx], device), acts)
            loss = -(weights[acts] * log_prob).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        tr_loss, tr_acc = evaluate(policy, X, y, device)
        msg = f"epoch {epoch:2}: train loss {tr_loss:.3f} acc {tr_acc:.0%}"
        if X_val is not None:
            va_loss, va_acc = evaluate(policy, X_val, y_val, device)
            msg += f" | held-out life: loss {va_loss:.3f} acc {va_acc:.0%}"
            best = min(best, (va_loss, epoch))
        print(msg)
    return model, best[1]


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lives = load_lives()
    if not lives:
        raise SystemExit("no recordings in demos/ yet: run record_demo.py first")
    data = [life_data(o, a, h) for o, a, h, _ in lives]
    keep = [i for i, (_, y) in enumerate(data) if len(y) >= 10]          # lives with something left to copy
    lives, data = [lives[i] for i in keep], [data[i] for i in keep]
    print(f"{len(lives)} lives used, {sum(len(y) for _, y in data)} steps (movement key, line >= {MIN_LINE}), "
          f"best line reached {max(b for _, _, _, b in lives)}")
    counts = np.bincount(np.concatenate([y for _, y in data]), minlength=len(m.ACTIONS))
    print("actions:", {m.ACTION_NAMES[i]: int(c) for i, c in enumerate(counts)})

    epochs = EPOCHS
    if len(lives) >= 2:
        # Hold out the highest life to see whether it generalises and how many epochs are enough.
        val = max(range(len(lives)), key=lambda i: lives[i][3])
        X_tr = np.concatenate([x for i, (x, _) in enumerate(data) if i != val])
        y_tr = np.concatenate([y for i, (_, y) in enumerate(data) if i != val])
        print(f"\nCheck run: training on {len(X_tr)} steps, holding out life {val} ({len(data[val][1])} steps)")
        _, epochs = train(X_tr, y_tr, EPOCHS, device, *data[val])
        epochs = max(1, epochs)
        print(f"best epoch on the held-out life: {epochs}")

    X_all = np.concatenate([x for x, _ in data])
    y_all = np.concatenate([y for _, y in data])
    print(f"\nFinal: training on all {len(X_all)} steps for {epochs} epochs")
    model, _ = train(X_all, y_all, epochs, device)
    model.save(OUT)
    print(f"saved {OUT}.zip -- next: test.py / play.py with MODEL = \"{OUT}\", then train.py with RESUME = \"{OUT}\"")
