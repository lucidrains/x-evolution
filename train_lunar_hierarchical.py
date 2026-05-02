# /// script
# dependencies = [
#     "fire",
#     "gymnasium[box2d]>=1.0.0",
#     "gymnasium[other]",
#     "x-evolution>=0.0.20",
#     "x-transformers",
#     "wandb"
# ]
# ///

import re
import fire
from shutil import rmtree
from itertools import cycle
from collections import deque

import numpy as np
import gymnasium as gym
import wandb
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F

from x_evolution import EvoStrategy
from x_transformers import Decoder

# helpers

def exists(v):
    return v is not None

# schedule parsing

def parse_string_schedule(schedule_str):
    schedule_str = re.sub(r'\(([^)]+)\)\s*\*\s*(\d+)', lambda m: f" {m.group(1)} " * int(m.group(2)), schedule_str)

    phases = []
    for duration, phase in re.findall(r'(\d+)\s*(all|head|trunk|tail)', schedule_str.lower()):
        phases.extend([phase] * int(duration))

    assert len(phases) > 0, 'could not parse phase schedule string'
    return phases

# orthogonal update from trunk to head

def orthogonal_project(x, residual):
    dtype = residual.dtype

    if x.device.type != 'mps':
        residual, x = residual.double(), x.double()

    unit = F.normalize(residual, dim = -1)
    parallel = (x * unit).sum(dim = -1, keepdim = True) * unit
    orthogonal = x - parallel

    return orthogonal.to(dtype)

# hierarchical transformer
# head -> trunk (residual gated every trunk_update_every steps) -> tail

class HierarchicalTransformer(Module):
    def __init__(
        self,
        dim_in,
        dim,
        num_actions,
        head_depth = 1,
        trunk_depth = 1,
        tail_depth = 1,
        trunk_update_every = 1
    ):
        super().__init__()
        self.trunk_update_every = trunk_update_every

        self.trunk_update_emb = nn.Parameter(torch.zeros(dim))

        self.token_emb = nn.Linear(dim_in, dim)

        decoder_kwargs = dict(
            dim = dim,
            attn_dim_head = 32,
            rotary_pos_emb = True,
            rotary_emb_dim = 32,
            pre_norm_has_final_norm = False
        )

        self.head = Decoder(depth = head_depth, **decoder_kwargs)

        self.state_to_trunk = nn.Linear(dim_in, dim)

        self.trunk = Decoder(depth = trunk_depth, **decoder_kwargs)
        self.trunk_gru_norm = nn.RMSNorm(dim)
        self.trunk_gru = nn.GRU(dim, dim, batch_first = True)

        self.tail = Decoder(depth = tail_depth, **decoder_kwargs)

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_actions)
        )

        self.trunk_action_gate = nn.Linear(dim, num_actions)

        self.trunk_to_logits = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Linear(dim, num_actions)
        )

    def forward(self, state, step = 0, cache = None):
        b, n, _ = state.shape
        assert n == 1, 'only single token rollouts are supported'

        x = self.token_emb(state)

        cache_head, cache_trunk, cache_tail, cache_gru, last_trunk_update, last_trunk_action = cache if exists(cache) else (None, None, None, None, None, None)

        # head

        x, cache_head = self.head(x, cache = cache_head, return_hiddens = True)

        should_update = (step % self.trunk_update_every) == 0

        # let the trunk network know when it is an update step

        x_trunk = x

        if should_update:
            x_trunk = x_trunk + self.trunk_update_emb

        # direct residual from state to trunk

        x_trunk = x_trunk + self.state_to_trunk(state)

        # trunk - always runs for KV context, but residual only applied every trunk_update_every steps

        x_trunk, cache_trunk = self.trunk(x_trunk, cache = cache_trunk, return_hiddens = True)

        x_trunk_gru, cache_gru = self.trunk_gru(self.trunk_gru_norm(x_trunk), cache_gru)
        x_trunk = x_trunk + x_trunk_gru

        if should_update:
            last_trunk_update = orthogonal_project(x_trunk, residual = x)
            last_trunk_action = self.trunk_to_logits(x_trunk)

        if exists(last_trunk_update):
            x = x + last_trunk_update

        # tail

        x, cache_tail = self.tail(x, cache = cache_tail, return_hiddens = True)

        logits = self.to_logits(x)

        if exists(last_trunk_action):
            gate = self.trunk_action_gate(x).sigmoid()
            logits = logits + gate * last_trunk_action

        return logits, (cache_head, cache_trunk, cache_tail, cache_gru, last_trunk_update, last_trunk_action)

# environment

class LunarEnvironment(Module):
    def __init__(
        self,
        video_folder = './recordings',
        render_every_eps = 500,
        max_steps = 500,
        repeats = 1,
        vectorized = False,
        num_envs = 1,
        rolling_window = 20
    ):
        super().__init__()
        self.vectorized = vectorized
        self.num_envs = num_envs

        if vectorized:
            env = gym.make_vec('LunarLander-v3', num_envs = num_envs, render_mode = 'rgb_array')
        else:
            env = gym.make('LunarLander-v3', render_mode = 'rgb_array')

        self.env = env
        self.max_steps = max_steps
        self.repeats = repeats
        self.video_folder = video_folder
        self.render_every_eps = render_every_eps
        self._pre_main_callback_called = False

        self.last_steps = deque(maxlen = rolling_window)

    def pre_main_callback(self):
        if self._pre_main_callback_called:
            return

        self._pre_main_callback_called = True
        rmtree(self.video_folder, ignore_errors = True)

        if not self.vectorized:
            self.env = gym.wrappers.RecordVideo(
                env = self.env,
                video_folder = self.video_folder,
                name_prefix = 'recording',
                episode_trigger = lambda eps_num: (eps_num % self.render_every_eps) == 0,
                disable_logger = True
            )

    @property
    def avg_steps(self):
        if len(self.last_steps) == 0:
            return 0.
        return sum(self.last_steps) / len(self.last_steps)

    def forward(self, model):
        device = next(model.parameters()).device
        seed = torch.randint(0, int(1e6), ())

        num_envs = self.num_envs if self.vectorized else 1
        cum_reward = torch.zeros(num_envs, device = device)

        for _ in range(self.repeats):
            state, _ = self.env.reset(seed = seed.item())

            step = 0
            dones = torch.zeros(num_envs, device = device, dtype = torch.bool)
            cache = None

            while step < self.max_steps and not dones.all():
                state_torch = torch.from_numpy(state).to(device)

                if not self.vectorized:
                    state_torch = state_torch.unsqueeze(0)

                state_torch = state_torch.unsqueeze(1)

                action_logits, cache = model(state_torch, step = step, cache = cache)
                action_logits = action_logits[:, -1, :]

                action = F.gumbel_softmax(action_logits, hard = True).argmax(dim = -1)

                env_action = action.detach().cpu().numpy() if self.vectorized else action.item()
                next_state, reward, truncated, terminated, *_ = self.env.step(env_action)

                reward_np = np.array(reward) if not isinstance(reward, np.ndarray) else reward
                total_reward = torch.from_numpy(reward_np).float().to(device)

                mask = (~dones).float()
                cum_reward += total_reward * mask

                dones_np = np.array(truncated | terminated) if not isinstance(truncated | terminated, np.ndarray) else (truncated | terminated)
                dones |= torch.from_numpy(dones_np).to(device)

                step += 1
                state = next_state

            self.last_steps.append(step)

        if not self.vectorized:
            return cum_reward.item() / self.repeats

        return cum_reward / self.repeats

# main

def main(
    vectorized = False,
    num_envs = 8,
    cpu = False,
    phase_schedule = '200all (50trunk 50tail)*10',
    trunk_update_every = 1,
    head_depth = 1,
    trunk_depth = 1,
    tail_depth = 1,
    dim = 64,
    use_wandb = True,
    wandb_project = 'lunar-hierarchical-transformer',
    rolling_window = 20,
    noise_population_size = 50,
    learning_rate = 1e-3,
    noise_scale = 1e-2
):
    if use_wandb:
        wandb.init(project = wandb_project, config = locals())

    model = HierarchicalTransformer(
        dim_in = 8,
        dim = dim,
        num_actions = 4,
        head_depth = head_depth,
        trunk_depth = trunk_depth,
        tail_depth = tail_depth,
        trunk_update_every = trunk_update_every
    )

    env = LunarEnvironment(
        repeats = 2,
        vectorized = vectorized,
        num_envs = num_envs,
        rolling_window = rolling_window
    )

    # partition parameters into head, trunk, tail

    head_params = list(model.token_emb.parameters()) + list(model.head.parameters())
    trunk_params = list(model.state_to_trunk.parameters()) + list(model.trunk.parameters()) + [model.trunk_update_emb] + list(model.trunk_gru.parameters()) + list(model.trunk_gru_norm.parameters()) + list(model.trunk_to_logits.parameters())
    tail_params = list(model.tail.parameters()) + list(model.to_logits.parameters()) + list(model.trunk_action_gate.parameters())

    head_param_ids = {id(p) for p in head_params}
    trunk_param_ids = {id(p) for p in trunk_params}
    tail_param_ids = {id(p) for p in tail_params}

    assert len(head_param_ids & trunk_param_ids) == 0, 'head and trunk parameters overlap'
    assert len(head_param_ids & tail_param_ids) == 0, 'head and tail parameters overlap'
    assert len(trunk_param_ids & tail_param_ids) == 0, 'trunk and tail parameters overlap'

    all_params = list(model.parameters())
    assert len(head_params) + len(trunk_params) + len(tail_params) == len(all_params), 'some parameters are not partitioned'

    evo_kwargs = dict(
        environment = env,
        vectorized = vectorized,
        vector_size = num_envs,
        cpu = cpu,
        num_generations = 1,
        noise_population_size = noise_population_size,
        noise_low_rank = 2,
        noise_scale = noise_scale,
        noise_scale_clamp_range = (5e-3, 2e-2),
        learned_noise_scale = True,
        use_sigma_optimizer = True,
        learning_rate = learning_rate,
        noise_scale_learning_rate = 1e-4,
        use_scheduler = False,
        verbose = False,
        sync_on_init = True
    )

    print('Setting up EvoStrategy wrappers...')

    evos = dict(
        all = EvoStrategy(model, params_to_optimize = all_params, **evo_kwargs),
        head = EvoStrategy(model, params_to_optimize = head_params, **evo_kwargs),
        trunk = EvoStrategy(model, params_to_optimize = trunk_params, **evo_kwargs),
        tail = EvoStrategy(model, params_to_optimize = tail_params, **evo_kwargs),
    )

    # schedule

    phases = parse_string_schedule(phase_schedule)
    total_generations = len(phases)
    phase_gen = cycle(phases)

    print('\n--- Training Phase Schedule ---')
    print(f'Schedule: {phase_schedule}')
    print(f'Total Generations: {total_generations}')
    print(f'Trunk updates every: {trunk_update_every} steps')
    print(f'Population Size: {noise_population_size}')
    print('-------------------------------\n')

    pbar = tqdm(total = total_generations, desc = 'Generations')
    running_rewards = deque(maxlen = rolling_window)

    for gen in range(1, total_generations + 1):
        phase = next(phase_gen)
        evo = evos[phase]

        fitnesses = evo(num_generations = 1, verbose = False)
        avg_fit = fitnesses.mean().item()

        running_rewards.append(avg_fit)
        avg_reward = sum(running_rewards) / len(running_rewards)
        avg_steps = env.avg_steps

        pbar.set_postfix(phase = phase, avg_reward = round(avg_reward, 2), avg_steps = round(avg_steps, 1))
        pbar.update(1)

        if use_wandb:
            wandb.log(dict(
                generation = gen,
                phase_str = phase,
                phase = {'all': 0, 'head': 1, 'trunk': 2, 'tail': 3}.get(phase, -1),
                avg_fitness = avg_fit,
                avg_reward_window = avg_reward,
                avg_steps = avg_steps,
            ))

    if use_wandb:
        wandb.finish()

if __name__ == '__main__':
    fire.Fire(main)
