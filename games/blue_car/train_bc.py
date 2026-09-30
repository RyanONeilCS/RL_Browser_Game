"""Behavior cloning: train the agent to copy Ryan's recorded driving (demos/*.npz from record_demo.py).

    python games/blue_car/train_bc.py

Result: models/bc_v1.zip, a normal PPO model (same network as train.py), so it can be tested with test.py /
play.py and improved further with PPO (train.py with RESUME = "models/bc_v1").

What it does:
1. Loads every recording and splits it into runs (restart = the cube counter jumps back to 0).
2. Keeps only good driving: drops the "YOU WIN / YOU LOSE" screen time, and the last ~2 s of runs that
   didn't reach the finish (the lead-up to a crash).
3. Stacks the last 4 observations exactly like VecFrameStack in training (zeros before a run's first step).
3b. Relabels steering by intent: Ryan steers with short taps (gas, gas+right, gas, gas+right, ...), so even in
   a curve most single steps are "gas" and a copy learns "always gas" (bc_v1 did: lost at the first curve every
   time). Each step gets the steering that dominates the few steps around it, and steering is weighted up.
4. Trains PPO's policy network to predict Ryan's action from the stacked pictures (cross-entropy, i.e.
   maximise the log-probability of his action), holding one run out to check it generalises.
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))   # demos/ and models/ are relative to this game's folder

import glob

import cv2
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

import blue_car_env as m

OUT = "models/bc_v4"
N_STACK = 4               # must match train.py (VecFrameStack n_stack)
CRASH_TAIL = 12           # steps dropped at the end of a run that didn't reach the finish (~2 s)
EPOCHS = 40
SMOOTH = 2                # relabel window: this many steps before and after (5 steps ~ 0.8 s)
MIN_STEER = 2             # steer in the window at least this many times (more than the other way) -> steer
BATCH = 64
LR = 3e-4


# ---- 1. load and split the recordings --------------------------------------------------------------

def restarts_from_frames(frames_dir, n_steps):
    """Steps where the cube counter shows 0 again after not being 0 = restarts (frames are every 3 steps)."""
    files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    if not files:
        return []
    digit = lambda f: (cv2.imread(f)[470:572, 10:200] > 200).all(axis=2)
    # The counter's "0", from a recorded frame checked by eye (not the recording's first frame: the game can still
    # show the previous run's score there, which made every "5" look like a restart).
    zero = cv2.imread("zero_digit.png", cv2.IMREAD_GRAYSCALE) > 0
    starts, was_zero = [], True
    for f in files:
        d = digit(f)
        # "0" differs from the start's "0" by 0-6 pixels, other digits by 1,000+. But "10" has its "0" in the same
        # place: it also matches, so the amount of white must match too (one digit, not two).
        is_zero = (d != zero).sum() < 150 and abs(int(d.sum()) - int(zero.sum())) < 100
        step = int(os.path.basename(f)[:5])
        if is_zero and not was_zero and step < n_steps:
            starts.append(step)
        was_zero = is_zero
    return starts


def end_screens_from_frames(frames_dir, n_steps):
    """True for steps where the YOU WIN / YOU LOSE screen shows (checked on the saved frames, every 3 steps)."""
    shown = np.zeros(n_steps, bool)
    env = m.BlueCarEnv.__new__(m.BlueCarEnv)       # only its screen checks, no browser
    for f in sorted(glob.glob(os.path.join(frames_dir, "*.jpg"))):
        step = int(os.path.basename(f)[:5])
        if step < n_steps and env._end_text(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)):
            shown[max(0, step - 2): step + 3] = True
    return shown


def load_runs():
    """List of (obs, actions, reached_finish) per run, with screen time and crash lead-ups removed."""
    runs = []
    for path in sorted(glob.glob("demos/*.npz")):
        d = np.load(path)
        obs, act = d["obs"], d["actions"]
        end_text = d["det_lost"] if "det_lost" in d else np.zeros(len(act), bool)   # YOU LOSE (recorder's log)
        end_text = end_text | end_screens_from_frames(path[:-4] + "_frames", len(act))   # + YOU WIN / LOSE on frames
        starts = set(np.nonzero(d["episode_starts"])[0].tolist())
        starts |= set(restarts_from_frames(path[:-4] + "_frames", len(act)))
        merged = []                                 # the same restart seen by both (a few steps apart): keep one
        for s0 in sorted(starts | {0}):
            if not merged or s0 - merged[-1] > 6:
                merged.append(s0)
        bounds = merged + [len(act)]
        for a, b in zip(bounds[:-1], bounds[1:]):
            shown = np.nonzero(end_text[a:b])[0]
            if len(shown):                          # end screen: keep only the driving before it
                end, finished = a + shown[0], True
            else:                                   # no end screen: restarted after a crash / pole
                end, finished = max(a, b - CRASH_TAIL), False
            if end - a >= 10:
                runs.append((obs[a:end], act[a:end], finished))
        print(f"{os.path.basename(path)}: {len(act)} steps -> {len(bounds) - 1} runs")
    return runs


# ---- 2. stack frames like VecFrameStack -----------------------------------------------------------

def stack(obs):
    """(T, 84, 84, C) -> (T, 84, 84, N_STACK*C): oldest first, newest last, zeros before the run's start."""
    T, H, W, C = obs.shape
    padded = np.concatenate([np.zeros((N_STACK - 1, H, W, C), obs.dtype), obs])
    return np.concatenate([padded[k:k + T] for k in range(N_STACK)], axis=3)


# ---- 2b. relabel taps as steady steering --------------------------------------------------------

STEER_DIR = {0: 0, 1: 0, 2: -1, 3: 1, 4: -1, 5: 1, 6: 0}      # -1 left, +1 right
GAS = {0: False, 1: True, 2: True, 3: True, 4: False, 5: False, 6: False}


def relabel(act):
    """Each step: gas as pressed; steering = the direction pressed most around it (if at least MIN_STEER times)."""
    out = act.copy()
    for t in range(len(act)):
        window = act[max(0, t - SMOOTH): t + SMOOTH + 1]
        right = sum(STEER_DIR[a] == 1 for a in window)
        left = sum(STEER_DIR[a] == -1 for a in window)
        steer = 1 if right >= MIN_STEER and right > left else -1 if left >= MIN_STEER and left > right else 0
        gas = GAS[act[t]] or sum(GAS[a] for a in window) > len(window) // 2
        out[t] = {(True, 0): 1, (True, -1): 2, (True, 1): 3, (False, 0): 0, (False, -1): 4, (False, 1): 5}[(gas, steer)]
    return out


# ---- 3. a PPO model with the right spaces (no browser needed) --------------------------------------

class SpacesOnly(gym.Env):
    """Just BlueCarEnv's observation/action spaces, so PPO builds exactly the network train.py uses."""
    observation_space = gym.spaces.Box(0, 255, (84, 84, 2), np.uint8)
    action_space = gym.spaces.Discrete(len(m.ACTIONS))

    def reset(self, seed=None, options=None):
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, False, {}


def to_tensor(x, device):
    return torch.as_tensor(x.transpose(0, 3, 1, 2), device=device)   # channels first, like VecTransposeImage


def evaluate(policy, X, y, device):
    policy.set_training_mode(False)
    with torch.no_grad():
        losses, correct = [], 0
        for i in range(0, len(X), 256):
            obs = to_tensor(X[i:i + 256], device)
            acts = torch.as_tensor(y[i:i + 256], device=device)
            _, log_prob, _ = policy.evaluate_actions(obs, acts)
            losses.append(-log_prob.sum().item())
            correct += (policy.get_distribution(obs).distribution.probs.argmax(1) == acts).sum().item()
    return sum(losses) / len(X), correct / len(X)


def class_weights(y):
    """Rarer actions count more, so "always gas" isn't the easy answer (inverse frequency, square-rooted)."""
    counts = np.bincount(y, minlength=len(m.ACTIONS)).astype(float)
    w = np.where(counts > 0, (counts.sum() / np.maximum(counts, 1)) ** 0.5, 0.0)
    return w / w[counts > 0].mean()


def train(X, y, epochs, device, X_val=None, y_val=None):
    env = VecFrameStack(DummyVecEnv([SpacesOnly]), n_stack=N_STACK)
    model = PPO("CnnPolicy", env, n_steps=1024, verbose=0, device=device)
    policy = model.policy
    opt = torch.optim.Adam(policy.parameters(), lr=LR)
    rng = np.random.default_rng(0)
    best = (float("inf"), 0)
    weights = torch.as_tensor(class_weights(y), dtype=torch.float32, device=device)
    for epoch in range(1, epochs + 1):
        policy.set_training_mode(True)
        order = rng.permutation(len(X))
        for i in range(0, len(X), BATCH):
            idx = order[i:i + BATCH]
            acts = torch.as_tensor(y[idx], device=device)
            _, log_prob, _ = policy.evaluate_actions(to_tensor(X[idx], device), acts)
            loss = -(weights[acts] * log_prob).mean()   # weighted cross-entropy with Ryan's (relabelled) action
            opt.zero_grad()
            loss.backward()
            opt.step()
        tr_loss, tr_acc = evaluate(policy, X, y, device)
        msg = f"epoch {epoch:2}: train loss {tr_loss:.3f} acc {tr_acc:.0%}"
        if X_val is not None:
            va_loss, va_acc = evaluate(policy, X_val, y_val, device)
            msg += f" | held-out run: loss {va_loss:.3f} acc {va_acc:.0%}"
            best = min(best, (va_loss, epoch))
        print(msg)
    return model, best[1]


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runs = load_runs()
    print(f"{len(runs)} usable runs, {sum(len(a) for _, a, _ in runs)} steps, "
          f"{sum(f for _, _, f in runs)} reached the finish")
    X_runs = [(stack(o), relabel(a)) for o, a, _ in runs]
    sym = "cG<>lr-"
    print("relabel example (run 4, first 50 steps):")
    print("  pressed: " + "".join(sym[x] for x in runs[4][1][:50]))
    print("  label:   " + "".join(sym[x] for x in X_runs[4][1][:50]))
    counts = np.bincount(np.concatenate([a for _, a in X_runs]), minlength=len(m.ACTIONS))
    print("actions:", {i: int(c) for i, c in enumerate(counts)}, "(0 coast, 1 gas, 2 gas+L, 3 gas+R, 4 L, 5 R)")

    # Hold out one finished run to see whether it generalises and how many epochs are enough.
    val = max(i for i, (_, _, f) in enumerate(runs) if f)
    X_tr = np.concatenate([x for i, (x, _) in enumerate(X_runs) if i != val])
    y_tr = np.concatenate([a for i, (_, a) in enumerate(X_runs) if i != val])
    print(f"\nCheck run: training on {len(X_tr)} steps, holding out run {val} ({len(X_runs[val][1])} steps)")
    _, best_epoch = train(X_tr, y_tr, EPOCHS, device, *X_runs[val])
    print(f"best epoch on the held-out run: {best_epoch}")

    # Final model: all runs, as many epochs as worked best.
    X_all = np.concatenate([x for x, _ in X_runs])
    y_all = np.concatenate([a for _, a in X_runs])
    print(f"\nFinal: training on all {len(X_all)} steps for {best_epoch} epochs")
    model, _ = train(X_all, y_all, best_epoch, device)
    model.save(OUT)
    print(f"saved {OUT}.zip")
