#!/usr/bin/env python3
# /// script
# dependencies = [
#     "accelerate",
#     "beartype",
#     "einops",
#     "fire",
#     "matplotlib",
#     "numpy",
#     "torch",
#     "torch-einops-utils",
#     "tqdm",
# ]
# ///
"""Meta-evolutionary Baldwin with nested ES.

Outer CMA-ES evolves an innate linear readout plus the inner ES learning rate
and noise scale. Each lifetime is a short inner ES trajectory. Selection sees
the mean validation loss along that trajectory, but only the innate genome is
inherited. A Darwinian control runs the same outer CMA-ES with no lifetime.

By default two orthogonal targets alternate every `switch_every` outer
generations, which is where lifetime adaptation pays off.

    uv run train_meta_baldwin.py
    uv run train_meta_baldwin.py --switch_every 0
"""

import math
from pathlib import Path

import fire
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from x_evolution import CMAEvoStrategy, EvoStrategy

DIM = 8
LIFETIME_BATCH = 64
VAL_BATCH = 256
GENERATIONS = 60
OUTER_POPULATION = 8
INNER_GENERATIONS = 3
INNER_POPULATION = 4
OUTER_NOISE = 0.1
INIT_LR = 1e-2
INIT_SIGMA = 5e-2
MAX_LR = 1.
MAX_SIGMA = 8e-2

def whitened_batch(n, seed):
    generator = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(n, DIM, generator = generator))
    return q * math.sqrt(n)

def make_targets(seed, count):
    generator = torch.Generator().manual_seed(seed * 100 + 3)

    if count == 1:
        return [torch.randn(DIM, generator = generator)]

    targets = []

    for _ in range(count):
        target = torch.randn(DIM, generator = generator)

        for other in targets:
            target = target - (target @ other) / (other @ other) * other

        targets.append(target / target.norm() * math.sqrt(DIM))

    return targets

def make_data(x_life, x_val, target):
    return x_life, (x_life @ target).unsqueeze(-1), x_val, (x_val @ target).unsqueeze(-1)

def mse(model, x, y):
    with torch.no_grad():
        return F.mse_loss(model(x), y).item()

def hparams(genome):
    lr = math.exp(float(genome.log_lr.clamp(-8., math.log(MAX_LR))))
    sigma = math.exp(float(genome.log_sigma.clamp(-8., math.log(MAX_SIGMA))))
    return lr, sigma

class MetaGenome(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, DIM) * 0.1)
        self.log_lr = nn.Parameter(torch.tensor([[-math.log(1 / INIT_LR)]]))
        self.log_sigma = nn.Parameter(torch.tensor([[-math.log(1 / INIT_SIGMA)]]))

    def forward(self, x):
        return x @ self.weight.T

def lifetime(genome, data, checkpoint):
    x_life, y_life, x_val, y_val = data
    lr, sigma = hparams(genome)

    trajectory = []
    adapted = [None]

    def on_generation(_, __, model):
        trajectory.append(mse(model, x_val, y_val))
        adapted[0] = model.weight.detach().clone()

    def environment(model):
        return -mse(model, x_life, y_life)

    inner = EvoStrategy(
        genome,
        environment = environment,
        num_generations = INNER_GENERATIONS,
        noise_population_size = INNER_POPULATION,
        learning_rate = lr,
        noise_scale = sigma,
        mirror_sampling = True,
        params_to_optimize = [genome.weight],
        on_generation = on_generation,
        cpu = True,
        verbose = False,
        checkpoint_path = str(checkpoint)
    )

    inner(rollback_model_at_end = True)
    return trajectory, adapted[0]

def run_seed(seed, condition, generations, out_dir, switch_every):
    torch.manual_seed(seed)

    targets = make_targets(seed, 2 if switch_every else 1)
    x_life = whitened_batch(LIFETIME_BATCH, seed * 100 + 1)
    x_val = whitened_batch(VAL_BATCH, seed * 100 + 2)

    def data_for_generation(generation):
        index = 0 if not switch_every else ((generation - 1) // switch_every) % 2
        return make_data(x_life, x_val, targets[index]), index

    task = {}
    task['data'], task['phase'] = data_for_generation(1)

    genome = MetaGenome()
    rows = []

    def environment(model):
        data = task['data']

        if condition == 'darwin':
            return -mse(model, data[2], data[3])

        trajectory, _ = lifetime(model, data, out_dir / 'checkpoints' / f'inner-{condition}-seed{seed}')
        return -sum(trajectory) / len(trajectory)

    def on_generation(generation, fitnesses, model):
        data = task['data']
        phase = task['phase']
        innate = mse(model, data[2], data[3])
        lr, sigma = hparams(model)

        if condition == 'darwin':
            post, correction = innate, 0.
        else:
            trajectory, adapted = lifetime(model, data, out_dir / 'checkpoints' / f'diag-{condition}-seed{seed}')
            post = trajectory[-1]
            correction = (adapted - model.weight.detach()).norm().item()

        rows.append(dict(
            condition = condition,
            seed = seed,
            generation = generation,
            phase = phase,
            innate = innate,
            post = post,
            correction = correction
        ))

        if generation == 1 or generation % 10 == 0 or generation == generations:
            print(
                f'[{condition}][seed {seed}][{generation:03d}] '
                f'innate {innate:.3e} | post {post:.3e} | '
                f'correction {correction:.3e} | lr {lr:.3e} | sigma {sigma:.3e}'
            )

        task['data'], task['phase'] = data_for_generation(generation + 1)

    outer = CMAEvoStrategy(
        genome,
        environment = environment,
        num_generations = generations,
        population_size = OUTER_POPULATION,
        noise_scale = OUTER_NOISE,
        lora_rank = None,
        freeze_base = False,
        mirror_sampling = False,
        on_generation = on_generation,
        cpu = True,
        verbose = False,
        checkpoint_path = str(out_dir / 'checkpoints' / f'outer-{condition}-seed{seed}')
    )

    outer()
    return rows

def summarize(rows, switch_every):
    baldwin = [row for row in rows if row['condition'] == 'baldwin']
    darwin = [row for row in rows if row['condition'] == 'darwin']

    def mean(rows, key):
        return float(np.mean([row[key] for row in rows]))

    print('\n===== summary =====')

    if switch_every:
        for phase in sorted(set(row['phase'] for row in rows)):
            b = [row for row in baldwin if row['phase'] == phase]
            d = [row for row in darwin if row['phase'] == phase]
            b_post, d_innate = mean(b, 'post'), mean(d, 'innate')

            print(
                f'phase {phase}: baldwin innate {mean(b, "innate"):.3e} | '
                f'post {b_post:.3e} | correction {mean(b, "correction"):.3e} | '
                f'darwin innate {d_innate:.3e} | post/darwin {b_post / d_innate:.3f}'
            )
    else:
        for condition in ('baldwin', 'darwin'):
            cond = [row for row in rows if row['condition'] == condition]
            first = [row for row in cond if row['generation'] == 1]
            last = [row for row in cond if row['generation'] == cond[-1]['generation']]

            print(
                f'{condition}: innate {mean(first, "innate"):.3e} -> {mean(last, "innate"):.3e} | '
                f'post {mean(first, "post"):.3e} -> {mean(last, "post"):.3e} | '
                f'correction {mean(first, "correction"):.3e} -> {mean(last, "correction"):.3e}'
            )

    b_post, d_innate = mean(baldwin, 'post'), mean(darwin, 'innate')
    print(
        f'overall: baldwin post {b_post:.3e} | darwin innate {d_innate:.3e} | '
        f'post/darwin {b_post / d_innate:.3f}'
    )

def plot(rows, out_dir, switch_every):
    generations = sorted(set(row['generation'] for row in rows))
    fig, axes = plt.subplots(1, 2, figsize = (11, 4))

    def curve(condition, key):
        return np.array([
            np.mean([row[key] for row in rows if row['condition'] == condition and row['generation'] == generation])
            for generation in generations
        ])

    ax = axes[0]
    ax.plot(generations, curve('baldwin', 'innate'), label = 'Baldwin innate', color = '#1f77b4')
    ax.plot(generations, curve('baldwin', 'post'), label = 'Baldwin post-lifetime', color = '#d62728')
    ax.plot(generations, curve('darwin', 'innate'), label = 'Darwin innate', color = '#222222', linestyle = '--')
    ax.set(yscale = 'log', xlabel = 'outer generation', ylabel = 'validation MSE', title = 'Baldwin vs Darwin')
    ax.legend(frameon = False)
    ax.grid(alpha = 0.25)

    ax = axes[1]
    ax.plot(generations, curve('baldwin', 'correction'), label = 'lifetime correction', color = '#9467bd')
    ax.set(yscale = 'log', xlabel = 'outer generation', ylabel = '||adapted - innate||', title = 'lifetime correction')
    ax.legend(frameon = False)
    ax.grid(alpha = 0.25)

    if switch_every:
        for ax in axes:
            for generation in range(switch_every + 1, max(generations) + 1, switch_every):
                ax.axvline(generation - 0.5, color = '0.6', alpha = 0.2, linewidth = 0.8)

    fig.tight_layout()
    fig.savefig(out_dir / 'curves.png', dpi = 150)
    plt.close(fig)

def main(seeds = 3, generations = GENERATIONS, out_dir = 'results/meta_baldwin', switch_every = 5):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents = True, exist_ok = True)

    rows = []

    for condition in ('baldwin', 'darwin'):
        for seed in range(seeds):
            rows.extend(run_seed(seed, condition, generations, out_dir, switch_every))

    summarize(rows, switch_every)
    plot(rows, out_dir, switch_every)
    print(f'wrote {out_dir / "curves.png"}')

if __name__ == '__main__':
    fire.Fire(main)
