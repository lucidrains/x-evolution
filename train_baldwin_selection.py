#!/usr/bin/env python3
# /// script
# dependencies = ["fire", "torch", "matplotlib", "numpy", "pillow"]
# ///
"""Baldwinian selection on a hidden-goal corridor.

    uv run train_baldwin_selection.py demo

The genome is an actor-critic initialization. A lifetime is four episodes with
an alternating hidden goal; fitness is the return after a short A2C adaptation.
Baldwinian inheritance keeps the pre-lifetime weights, Lamarckian keeps the
adapted weights, and Darwinian has no lifetime.
"""
from __future__ import annotations

import copy
import csv
from pathlib import Path

import fire
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from torch import nn
import torch.nn.functional as F

HORIZON, DT, N_ACTIONS = 30, 0.1, 3
GOALS = {'broad': (1., -1., 1., -1.), 'narrow': (1., 1., 1., 1.)}
EVAL_GOALS = {'broad': (1., -1.), 'narrow': (1.,)}
COLORS = {'baldwin': '#1f77b4', 'lamarck': '#d62728', 'darwin': '#222222'}
LABELS = {'baldwin': 'Baldwinian', 'lamarck': 'Lamarckian', 'darwin': 'Darwinian'}

# helpers

def mean(xs):
    return sum(xs) / len(xs)

def resolve_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

def params_of(module):
    return {name: p.detach().clone() for name, p in module.named_parameters()}

def set_params(module, params):
    with torch.no_grad():
        for name, p in module.named_parameters():
            p.copy_(params[name])

# env

def step_env(x, action, goal):
    x = (x + (action.float() - 1.) * DT).clamp(-1., 1.)
    return x, 1. - (x - goal).abs()

# policy

class ActorCritic(nn.Module):
    def __init__(self, hidden = 16):
        super().__init__()
        self.torso = nn.Sequential(nn.Linear(1, hidden), nn.Tanh())
        self.pi = nn.Linear(hidden, N_ACTIONS)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.torso(x[..., None])
        return self.pi(h), self.v(h).squeeze(-1)

# lifetime

@torch.no_grad()
def evaluate(policy, goal, n = 64):
    device = next(policy.parameters()).device
    x = torch.zeros(n, device = device)
    total = torch.zeros_like(x)

    for _ in range(HORIZON):
        logits, _ = policy(x)
        x, reward = step_env(x, logits.argmax(dim = -1), goal)
        total += reward

    return total.mean().item()

def rollout(policy, goal, n):
    x = torch.zeros(n, device = next(policy.parameters()).device)
    logps, values, rewards, entropies = [], [], [], []

    for _ in range(HORIZON):
        logits, value = policy(x)
        dist = torch.distributions.Categorical(logits = logits)
        action = dist.sample()

        logps.append(dist.log_prob(action)); values.append(value); entropies.append(dist.entropy())
        x, reward = step_env(x, action, goal); rewards.append(reward)
    rewards = torch.stack(rewards, dim = 1)
    returns = rewards.flip(1).cumsum(1).flip(1)

    return torch.stack(logps, 1), torch.stack(values, 1), returns, torch.stack(entropies, 1)

def adapt(policy, goal, steps, rollouts, n, lr):
    policy = copy.deepcopy(policy)
    optimizer = torch.optim.Adam(policy.parameters(), lr = lr)

    for _ in range(steps):
        batch = [rollout(policy, goal, n) for _ in range(rollouts)]
        logps, values, returns, entropies = (torch.cat(xs, 0) for xs in zip(*batch))
        advantage = returns - values.detach()
        loss = -(logps * advantage).mean() + 0.5 * F.mse_loss(values, returns) - 0.01 * entropies.mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    return policy

def lifetime(policy, condition, goals, steps, rollouts, n, lr):
    returns, current = [], policy

    for goal in goals:
        if condition == 'baldwin':
            current = policy

        if condition != 'darwin':
            current = adapt(current, goal, steps, rollouts, n, lr)

        returns.append(evaluate(current, goal))

    return returns, params_of(current)

# evolution

def train(condition = 'baldwin', task = 'broad', seed = 0, generations = 20, population = 16,
          topk = 4, sigma = 0.1, sigma_decay = 0.99, update_lr = 0.5, steps = 2,
          rollouts = 1, episodes = 8, lr = 0.03, out_dir = 'results/baldwin_selection'):
    assert 0 < topk <= population and steps >= 0 and generations > 0

    torch.manual_seed(seed)
    policy = ActorCritic().to(resolve_device())

    base = params_of(policy); keys = list(base)
    goals, eval_goals = GOALS[task], EVAL_GOALS[task]
    out = Path(out_dir); out.mkdir(parents = True, exist_ok = True)
    rows = []

    for generation in range(1, generations + 1):
        sigma_t = sigma * sigma_decay ** (generation - 1)
        candidates = [
            {key: base[key] + sigma_t * torch.randn_like(value) for key, value in base.items()}
            for _ in range(population)
        ]
        fits, adapts = [], []

        for candidate in candidates:
            set_params(policy, candidate)

            if condition == 'darwin':
                fit, adapted = mean([evaluate(policy, goal) for goal in eval_goals]), candidate
            else:
                returns, adapted = lifetime(policy, condition, goals, steps, rollouts, episodes, lr)
                fit = mean(returns)

            fits.append(fit)
            adapts.append(adapted)

        selected = sorted(range(population), key = lambda i: fits[i], reverse = True)[:topk]
        inherited = [adapts[i] if condition == 'lamarck' else candidates[i] for i in selected]
        mean_inherited = {key: torch.stack([vec[key] for vec in inherited]).mean(dim = 0) for key in keys}
        base = {key: base[key] + update_lr * (mean_inherited[key] - base[key]) for key in keys}

        set_params(policy, base)
        innate = mean([evaluate(policy, goal) for goal in eval_goals])
        rows.append(dict(generation = generation, condition = condition, task = task, seed = seed,
                         fitness = mean(fits), innate = innate))
        print(f'[{generation:03d}] fitness {rows[-1]["fitness"]:5.2f} | innate {innate:5.2f}')

    with (out / f'{condition}_{task}_seed{seed}.csv').open('w', newline = '') as f:
        writer = csv.DictWriter(f, fieldnames = list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    config = dict(condition = condition, task = task, seed = seed, generations = generations, population = population,
                  topk = topk, sigma = sigma, sigma_decay = sigma_decay, update_lr = update_lr,
                  steps = steps, rollouts = rollouts, episodes = episodes, lr = lr)

    torch.save(dict(state_dict = {key: value.cpu() for key, value in base.items()}, config = config),
               out / f'{condition}_{task}_seed{seed}.pt')

# plots

def load_rows(out_dir, task):
    rows = []
    for path in sorted(Path(out_dir).glob(f'*_{task}_seed*.csv')):
        with path.open() as f:
            rows.extend(csv.DictReader(f))
    return rows

def plot_curves(out_dir = 'results/baldwin_selection', task = 'broad'):
    out = Path(out_dir)
    rows = load_rows(out, task)
    assert rows, f'no runs in {out}'

    fig, ax = plt.subplots(figsize = (7, 4))
    for condition in ('baldwin', 'lamarck', 'darwin'):
        cond_rows = [r for r in rows if r['condition'] == condition]
        if not cond_rows:
            continue
        gens = sorted({int(r['generation']) for r in cond_rows})
        fitness = [mean([float(r['fitness']) for r in cond_rows if int(r['generation']) == g]) for g in gens]
        ax.plot(gens, fitness, color = COLORS[condition], lw = 2, label = LABELS[condition])

    ax.set(title = f'lifetime fitness on a hidden-goal corridor ({task})', xlabel = 'generation', ylabel = 'return')
    ax.legend(frameon = False); ax.grid(alpha = 0.25); ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(); fig.savefig(out / 'curves.png', dpi = 160, bbox_inches = 'tight'); plt.close(fig)
    print(f'wrote {out / "curves.png"}')

def load_policy(path):
    ckpt = torch.load(path, map_location = 'cpu', weights_only = False)
    policy = ActorCritic()
    policy.load_state_dict(ckpt['state_dict'])
    return policy, ckpt['config']

def trajectory(policy, goal):
    x, xs = 0., []

    for _ in range(HORIZON):
        with torch.no_grad():
            logits, _ = policy(torch.zeros(1))

        x, _ = step_env(torch.tensor([x]), logits.argmax(dim = -1), goal)
        xs.append(x.item())

    return np.array(xs)

def adapt_with(policy, goal, config):
    return adapt(policy, goal, config['steps'], config['rollouts'], config['episodes'], config['lr'])

def lamarck_policy(policy, config):
    for goal in GOALS[config['task']]:
        policy = adapt_with(policy, goal, config)
    return policy

def trajectories_and_video(out_dir = 'results/baldwin_selection', task = 'broad', seed = 0, fps = 10):
    out = Path(out_dir)
    baldwin, bc = load_policy(out / f'baldwin_{task}_seed{seed}.pt')
    lamarck, lc = load_policy(out / f'lamarck_{task}_seed{seed}.pt')
    lamarck_final = lamarck_policy(lamarck, lc)

    fig, axes = plt.subplots(2, 2, figsize = (10, 7), sharex = True, sharey = True)
    dots = []

    for row, goal in enumerate((1., -1.)):
        panels = (('Baldwin', baldwin, adapt_with(baldwin, goal, bc)), ('Lamarck', lamarck, lamarck_final))
        for col, (name, innate, post) in enumerate(panels):
            ax = axes[row, col]
            innate_xs = trajectory(innate, goal)
            post_xs = trajectory(post, goal)
            color = COLORS[name.lower()]
            ax.plot(range(1, HORIZON + 1), innate_xs, color = '#999999', ls = '--', lw = 2, label = 'innate')
            ax.plot(range(1, HORIZON + 1), post_xs, color = color, lw = 2, label = 'post-learning')
            ax.axhline(goal, color = '#2ca02c', ls = ':', lw = 1)
            dot, = ax.plot([1], [post_xs[0]], 'o', color = color, ms = 8, zorder = 4)
            cursor = ax.axvline(1, color = color, ls = ':', lw = 1, alpha = 0.5)
            dots.append((dot, cursor, post_xs))
            ax.set(title = f'{name} | goal {goal:+.0f}', ylim = (-1.05, 1.05))
            ax.grid(alpha = 0.25); ax.legend(frameon = False, fontsize = 8)

    fig.supxlabel('step'); fig.supylabel('position x'); fig.suptitle('innate vs post-learning trajectories', y = 0.98)
    for dot, cursor, _ in dots: dot.set_visible(False); cursor.set_visible(False)
    fig.tight_layout(rect = (0, 0, 1, 0.96))
    fig.savefig(out / 'trajectories.png', dpi = 160, bbox_inches = 'tight')
    for dot, cursor, _ in dots: dot.set_visible(True); cursor.set_visible(True)

    def update(frame):
        for dot, cursor, xs in dots:
            dot.set_data([frame + 1], [xs[frame]])
            cursor.set_xdata([frame + 1, frame + 1])
        return [artist for dot, cursor, _ in dots for artist in (dot, cursor)]

    anim = FuncAnimation(fig, update, frames = HORIZON, interval = 1000 / fps, blit = False)
    anim.save(out / 'baldwin_vs_lamarck.gif', writer = PillowWriter(fps = fps))

    try:
        anim.save(out / 'baldwin_vs_lamarck.mp4', writer = FFMpegWriter(fps = fps, bitrate = 2400))
    except Exception as e:
        print(f'mp4 skipped: {e}')

    plt.close(fig)
    print(f'wrote {out / "trajectories.png"} and {out / "baldwin_vs_lamarck.gif"}')

# demo

def demo(seeds = 1, generations = 20, out_dir = 'results/baldwin_selection'):
    for condition in ('baldwin', 'lamarck', 'darwin'):
        for seed in range(seeds):
            train(condition, 'broad', seed, generations, out_dir = out_dir)

    plot_curves(out_dir)
    trajectories_and_video(out_dir)

if __name__ == '__main__':
    fire.Fire(dict(train = train, demo = demo, plot = plot_curves,
                   trajectories = trajectories_and_video, video = trajectories_and_video))
