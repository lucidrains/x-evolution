# /// script
# dependencies = [
#     "einops",
#     "fire",
#     "gymnasium[box2d]>=1.0.0",
#     "gymnasium[other]",
#     "imageio[ffmpeg]",
#     "numpy",
#     "torch",
#     "torch-einops-utils>=0.2.4",
#     "wandb",
#     "x-evolution>=0.2.1"
# ]
# ///

from __future__ import annotations

import os
import re
import math
import time
from pathlib import Path
from collections import deque, namedtuple

import fire
import imageio.v2 as imageio
import numpy as np
import gymnasium as gym
import wandb

import torch
from torch import nn, Tensor, tensor, zeros, full
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Parameter
import torch.distributed as dist

from einops import repeat, einsum, rearrange
from einops.layers.torch import Rearrange

from torch_einops_utils import (
    cast_tensor,
    clamp,
    entropy,
    pack_with_inverse,
    pad_right_at_dim,
    safe_stack,
    temp_eval
)
from torch_einops_utils.device import module_device

from accelerate import Accelerator

from x_evolution import CMAEvoStrategy

torch.set_num_threads(1)

# types

Memory = namedtuple('Memory', (
    'step',
    'latents',
    'chunks'
))

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def log(t, eps = 1e-20):
    return t.clamp_min(eps).log()

def resolve_hierarchies(hierarchies: tuple[int, ...] | int) -> tuple[int, ...]:
    if isinstance(hierarchies, int):
        return tuple(2 ** i for i in range(hierarchies))
    return tuple(int(stride) for stride in hierarchies)

# combined hierarchical auxiliary loss

def hierarchical_auxiliary_loss(
    all_code_probs: Tensor,
    pos_code_probs: Tensor | None = None,
    pos_cf_action_probs: Tensor | None = None,
    pos_target_actions: Tensor | None = None,
    coupling_loss_weight = 1.5,
    latent_mi_weight = 3.0
):
    zero = tensor(0., device = all_code_probs.device, requires_grad = True)

    # normalized mutual info between states and latent codes
    # encourages diverse prototype usage across trajectory states while keeping per-state selection decisive

    num_classes = all_code_probs.shape[-1]

    if num_classes > 1 and all_code_probs.shape[-2] > 1:
        marginal = all_code_probs.mean(dim = -2)
        latent_mi = (entropy(marginal) - entropy(all_code_probs)) / math.log(num_classes)
    else:
        latent_mi = zero

    # composed counterfactual action coupling loss on above-average episodes
    # marginalizes counterfactual action probs over higher-level code probs to match taken actions

    if exists(pos_code_probs) and exists(pos_cf_action_probs) and exists(pos_target_actions):
        composed_probs = einsum(pos_code_probs, pos_cf_action_probs, 'n k, n k a -> n a')
        couple_loss = F.nll_loss(log(composed_probs), pos_target_actions.long())
    else:
        couple_loss = zero

    return coupling_loss_weight * couple_loss - latent_mi_weight * latent_mi

# schedule parsing

def parse_string_schedule(schedule_str):
    schedule_str = re.sub(r'\(([^)]+)\)\s*\*\s*(\d+)', lambda m: f' {m.group(1)} ' * int(m.group(2)), schedule_str)

    phases = [
        (phase, int(duration))
        for duration, phase in re.findall(r'(\d+)\s*(all|level_\d+)', schedule_str.lower())
    ]

    assert len(phases) > 0, 'could not parse phase schedule string'
    return phases

# 2d affine linear layer so cma-es jointly evolves weights and biases

class AffineLinear(Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        bias_scale = 0.25,
        init_gain = None
    ):
        super().__init__()
        self.bias_scale = bias_scale
        self.weight = Parameter(torch.empty(dim_out, dim_in + 1))

        if exists(init_gain):
            nn.init.orthogonal_(self.weight[:, :-1], gain = init_gain)
            nn.init.zeros_(self.weight[:, -1])
        else:
            nn.init.kaiming_uniform_(self.weight, a = 5 ** 0.5)

    def forward(self, x):
        return F.linear(pad_right_at_dim(x, 1, value = self.bias_scale), self.weight)

# simba residual block - https://arxiv.org/abs/2410.09754

class SimBaBlock(Module):
    def __init__(
        self,
        dim,
        expansion_factor = 1,
        dropout = 0.
    ):
        super().__init__()
        dim_inner = int(dim * expansion_factor)

        self.net = nn.Sequential(
            nn.RMSNorm(dim),
            AffineLinear(dim, dim_inner),
            nn.SiLU(),
            nn.Dropout(dropout),
            AffineLinear(dim_inner, dim, init_gain = 0.2)
        )

    def forward(self, x):
        return x + self.net(x)

# discrete latent readout and hypernetwork decoder

class DiscreteLatentReadout(Module):
    def __init__(
        self,
        dim,
        dim_latent,
        num_discrete = 4,
        temperature = 0.7,
        logit_cap = 3.
    ):
        super().__init__()
        self.num_discrete = num_discrete
        self.temperature = temperature
        self.logit_cap = logit_cap

        self.norm = nn.RMSNorm(dim)
        self.to_logits = AffineLinear(dim, num_discrete, init_gain = 0.5)

        self.prototypes = Parameter(torch.empty(num_discrete, dim_latent))
        nn.init.orthogonal_(self.prototypes, gain = (dim_latent ** 0.5) * 0.5)

    def forward(self, x):
        raw_logits = self.to_logits(self.norm(x))
        logits = self.logit_cap * torch.tanh(raw_logits / self.logit_cap)

        probs = (logits / self.temperature).softmax(dim = -1)
        indices = probs.argmax(dim = -1)
        latents = self.prototypes[indices]

        return latents, probs

class HyperNetworkDecoder(Module):
    def __init__(
        self,
        dim,
        dim_latent,
        low_rank = 2,
        num_time_embeds = 4
    ):
        super().__init__()
        self.num_time_embeds = max(1, num_time_embeds)
        self.time_emb = nn.Embedding(self.num_time_embeds, dim)
        self.latent_time_emb = nn.Embedding(self.num_time_embeds, dim_latent)
        nn.init.normal_(self.time_emb.weight, std = 0.02)
        nn.init.normal_(self.latent_time_emb.weight, std = 0.02)

        self.decoder = nn.Sequential(
            nn.RMSNorm(dim_latent),
            AffineLinear(dim_latent, dim_latent * 2),
            nn.SiLU(),
            nn.Linear(dim_latent * 2, 2 * low_rank * dim, bias = False)
        )

        weight = self.decoder[-1].weight
        half = weight.shape[0] // 2
        nn.init.orthogonal_(weight[:half], gain = 1.)
        nn.init.zeros_(weight[half:])

        self.to_hyper_network_weights = Rearrange('... (two d r) -> two ... d r', two = 2, r = low_rank)

    def forward(self, x, latent, time_indices = None):
        if exists(time_indices):
            clamped_indices = clamp(time_indices, 0, self.num_time_embeds - 1)
            x = x + self.time_emb(clamped_indices)
            latent = latent + self.latent_time_emb(clamped_indices)

        weights_in, weights_out = self.to_hyper_network_weights(self.decoder(latent))
        return einsum(x, weights_in, weights_out, '... i, ... i r, ... o r -> ... o')

# composable hierarchy level

class HierarchyLevel(Module):
    def __init__(
        self,
        state_dim,
        dim,
        dim_latent,
        *,
        is_bottom = False,
        is_top = False,
        num_actions = 4,
        num_internal_actions = 4,
        low_rank = 2,
        num_time_embeds = 4,
        dropout = 0.
    ):
        super().__init__()
        self.is_bottom = is_bottom
        self.is_top = is_top

        self.proj_in = AffineLinear(state_dim, dim)
        self.stem = SimBaBlock(dim, dropout = dropout)

        if not is_top:
            self.from_higher = HyperNetworkDecoder(
                dim,
                dim_latent,
                low_rank = low_rank,
                num_time_embeds = num_time_embeds
            )

        self.upper = SimBaBlock(dim, dropout = dropout)

        if is_bottom:
            self.action_head = nn.Sequential(
                AffineLinear(dim, dim // 2),
                nn.SiLU(),
                nn.Linear(dim // 2, num_actions, bias = False)
            )
            nn.init.orthogonal_(self.action_head[-1].weight, gain = 0.5)
        else:
            self.readout = DiscreteLatentReadout(dim, dim_latent, num_discrete = num_internal_actions)
            self.residual_proj = nn.Linear(dim, dim_latent, bias = False)
            nn.init.zeros_(self.residual_proj.weight)
            self.register_buffer('gate', tensor(0.))

    def encode_stem(self, state):
        return self.stem(self.proj_in(state))

    def forward(
        self,
        state,
        latent_from_higher = None,
        time_indices = None
    ):
        hidden = self.encode_stem(state)

        if not self.is_top and exists(latent_from_higher):
            hidden = hidden + self.from_higher(hidden, latent_from_higher, time_indices = time_indices)

        upper_hidden = self.upper(hidden)

        if self.is_bottom:
            return self.action_head(upper_hidden)

        latent, probs = self.readout(upper_hidden)
        latent = self.gate * (latent + self.residual_proj(upper_hidden))
        return latent, probs

# hierarchical actor

class HierarchicalActor(Module):
    def __init__(
        self,
        state_dim = 8,
        hidden_dim = 4,
        num_actions = 4,
        num_internal_actions = 4,
        dim_latent = 4,
        low_rank = 1,
        hierarchies: tuple[int, ...] = (1, 4),
        phase_shifts: tuple[int, ...] | None = None,
        coupling_loss_weight = 1.5,
        latent_mi_weight = 3.0,
        dropout = 0.
    ):
        super().__init__()
        assert len(hierarchies) >= 1 and hierarchies[0] == 1, 'hierarchies must start with base stride 1, e.g. (1, 4)'

        self.state_dim = state_dim
        self.num_actions = num_actions
        self.num_internal_actions = num_internal_actions
        self.dim_latent = dim_latent
        self.hierarchies = tuple(hierarchies)
        self.num_levels = len(self.hierarchies)

        self.coupling_loss_weight = coupling_loss_weight
        self.latent_mi_weight = latent_mi_weight

        phase_shifts = default(phase_shifts, (0,) * self.num_levels)
        assert len(phase_shifts) == self.num_levels
        self.phase_shifts = tuple(phase_shifts)

        dim = hidden_dim * 2
        levels = []

        for level_index in range(self.num_levels):
            is_bottom = (level_index == 0)
            is_top = (level_index == self.num_levels - 1)
            num_time_embeds = max(1, self.hierarchies[level_index + 1] // self.hierarchies[level_index]) if not is_top else 1

            levels.append(HierarchyLevel(
                state_dim, dim, dim_latent,
                is_bottom = is_bottom, is_top = is_top,
                num_actions = num_actions, num_internal_actions = num_internal_actions,
                low_rank = low_rank, num_time_embeds = num_time_embeds, dropout = dropout
            ))

        self.levels = ModuleList(levels)

    def enable_level_(self, level_index: int):
        if 0 < level_index < self.num_levels:
            self.levels[level_index].gate.fill_(1.)

    def level_parameters(self):
        groups = dict(all = list(self.parameters()))

        for level_index, level in enumerate(self.levels):
            params = [*level.proj_in.parameters(), *level.stem.parameters(), *level.upper.parameters()]

            if level.is_bottom:
                params.extend(level.action_head.parameters())
            else:
                params.extend([
                    *level.readout.parameters(),
                    *level.residual_proj.parameters(),
                    *self.levels[level_index - 1].from_higher.parameters()
                ])

            groups[f'level_{level_index}'] = params

        return groups

    def level_time_indices(self, level_index: int, timesteps: Tensor | int | None):
        if not exists(timesteps) or level_index >= self.num_levels - 1:
            return None

        stride_current = self.hierarchies[level_index]
        stride_higher = self.hierarchies[level_index + 1]
        shift_higher = self.phase_shifts[level_index + 1]
        return ((timesteps - shift_higher) % stride_higher) // stride_current

    def counterfactual_action_probs(self, state, level_index = 0, time_indices = None, temperature = 0.5):
        assert 0 <= level_index < self.num_levels - 1

        state, _ = pack_with_inverse(state, '* d')

        num_codes = self.num_internal_actions
        receiver = self.levels[level_index]
        sender = self.levels[level_index + 1]

        hidden = receiver.encode_stem(state).detach()
        base_logits = receiver.action_head(receiver.upper(hidden))

        repeated_base_logits = repeat(base_logits, 'b a -> (b k) a', k = num_codes)
        repeated_hidden = repeat(hidden, 'b d -> (b k) d', k = num_codes)
        repeated_time_indices = repeat(time_indices, 'b -> (b k)', k = num_codes) if exists(time_indices) else None
        repeated_prototypes = repeat(sender.readout.prototypes, 'k d -> (b k) d', b = hidden.shape[0])

        steered = repeated_hidden + receiver.from_higher(repeated_hidden, repeated_prototypes, time_indices = repeated_time_indices)
        steered_logits = receiver.action_head(receiver.upper(steered))

        delta_probs = ((steered_logits - repeated_base_logits) / temperature).softmax(dim = -1)
        return rearrange(delta_probs, '(b k) a -> b k a', k = num_codes)

    def aux_loss(
        self,
        memory: Memory | None,
        episode_returns: Tensor
    ):
        device = episode_returns.device
        zero = tensor(0., device = device)

        if not exists(memory) or len(memory.chunks) <= 1:
            return zero

        states, probs, actions, masks = map(safe_stack, zip(*memory.chunks))

        all_code_probs = probs[masks]

        if all_code_probs.shape[0] <= 1:
            return zero

        pos_mask = masks & (episode_returns >= episode_returns.mean())

        pos_probs, pos_cf_probs, pos_actions = None, None, None

        if pos_mask.any():
            pos_states = states[pos_mask][:64]
            pos_probs = probs[pos_mask][:64]
            pos_actions = actions[pos_mask][:64]

            pos_cf_probs = self.counterfactual_action_probs(
                pos_states,
                level_index = 0,
                time_indices = zeros(pos_states.shape[0], dtype = torch.long, device = device)
            )

        return hierarchical_auxiliary_loss(
            all_code_probs,
            pos_probs,
            pos_cf_probs,
            pos_actions,
            coupling_loss_weight = self.coupling_loss_weight,
            latent_mi_weight = self.latent_mi_weight
        )

    def forward(
        self,
        state: Tensor,
        memory: Memory | None = None,
        dones: Tensor | None = None
    ):
        state, inverse_pack = pack_with_inverse(state, '* d')
        batch, device = state.shape[0], state.device

        if not exists(memory):
            step = 0
            latents = [None] * self.num_levels
            chunks = []
        else:
            step, latents, chunks = memory
            latents = [*latents]
            chunks = [*chunks]

        timestep = full((batch,), step, dtype = torch.long, device = device)
        level_1_probs = None

        for level_index in reversed(range(1, self.num_levels)):
            level = self.levels[level_index]

            if level.gate.item() <= 0.:
                continue

            stride = self.hierarchies[level_index]
            shift = self.phase_shifts[level_index]

            if step == 0 or divisible_by(step - shift, stride):
                time_indices = self.level_time_indices(level_index, timestep)
                latent_above = latents[level_index + 1] if (level_index + 1) < self.num_levels else None

                latents[level_index], probs = level(
                    state,
                    latent_from_higher = latent_above,
                    time_indices = time_indices
                )

                if level_index == 1:
                    level_1_probs = probs

        latent_from_higher = latents[1] if self.num_levels > 1 else None
        base_time_indices = self.level_time_indices(0, timestep) if exists(latent_from_higher) else None

        action_logits = self.levels[0](
            state,
            latent_from_higher = latent_from_higher,
            time_indices = base_time_indices
        )

        if exists(level_1_probs):
            actions = action_logits.argmax(dim = -1)
            active_mask = ~dones if exists(dones) else torch.ones(batch, dtype = torch.bool, device = device)
            chunks.append((state, level_1_probs, actions, active_mask))

        return inverse_pack(action_logits), Memory(step + 1, latents, chunks)

# environment

class LunarEnvironment(Module):
    def __init__(
        self,
        env_name = 'LunarLander-v3',
        num_envs = 2,
        max_steps = 500,
        eval_seeds: tuple[int, ...] = tuple(range(101, 113)),
        rolling_window = 20
    ):
        super().__init__()
        self.env_name = env_name
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.eval_seeds = tuple(eval_seeds)

        max_pool = max(num_envs, len(self.eval_seeds))
        self.envs = [gym.make(env_name, max_episode_steps = max_steps) for _ in range(max_pool)]
        self.last_steps = deque(maxlen = rolling_window)

    @property
    def avg_steps(self):
        if len(self.last_steps) == 0:
            return 0.
        return sum(self.last_steps) / len(self.last_steps)

    @torch.no_grad()
    def rollout(self, actor: HierarchicalActor, seeds = None):
        device = module_device(actor)
        seeds = default(seeds, self.eval_seeds[:self.num_envs])
        num_active_envs = len(seeds)

        obs_list = [self.envs[i].reset(seed = int(seeds[i]))[0] for i in range(num_active_envs)]
        state = np.stack(obs_list, axis = 0)

        cum_reward = np.zeros(num_active_envs, dtype = np.float32)
        dones = np.zeros(num_active_envs, dtype = bool)
        steps_taken = np.zeros(num_active_envs, dtype = np.int32)

        memory = None

        for _ in range(self.max_steps):
            if dones.all():
                break

            state_tensor = cast_tensor(state, dtype = torch.float32, device = device)
            dones_tensor = cast_tensor(dones, dtype = torch.bool, device = device)

            action_logits, memory = actor(state_tensor, memory = memory, dones = dones_tensor)
            actions = action_logits.argmax(dim = -1).tolist()

            for env_idx in np.flatnonzero(~dones):
                next_obs, reward, terminated, truncated, _ = self.envs[env_idx].step(actions[env_idx])
                cum_reward[env_idx] += float(reward)
                steps_taken[env_idx] += 1

                if terminated or truncated:
                    dones[env_idx] = True
                else:
                    state[env_idx] = next_obs

        self.last_steps.append(float(steps_taken.mean()))

        env_return = float(cum_reward.mean())
        ep_returns = cast_tensor(cum_reward, dtype = torch.float32, device = device)
        aux_loss = actor.aux_loss(memory, ep_returns)
        fitness = env_return - float(aux_loss.item())

        return fitness, env_return, cum_reward

    def evaluate(self, actor: HierarchicalActor):
        return self.rollout(actor, seeds = self.eval_seeds)

    def forward(self, model):
        actor = model.model if hasattr(model, 'model') else model
        seeds = torch.randint(0, int(1e6), (self.num_envs,)).tolist()
        fitness, _, _ = self.rollout(actor, seeds = seeds)
        return fitness

# evaluation and video recording

@torch.no_grad()
def record_best_episode_video(
    actor: HierarchicalActor,
    env_name: str = 'LunarLander-v3',
    video_folder: str = './recordings_hierarchical',
    video_prefix: str = 'hierarchical-best',
    max_timesteps: int = 500,
    eval_seeds: tuple[int, ...] = tuple(range(101, 113))
):
    device = module_device(actor)
    env = gym.make(env_name, render_mode = 'rgb_array', max_episode_steps = max_timesteps)

    best_return = -float('inf')
    best_frames = []
    best_seed = eval_seeds[0]

    with temp_eval(actor):
        for episode_seed in eval_seeds:
            obs, _ = env.reset(seed = int(episode_seed))
            state = cast_tensor(obs, dtype = torch.float32, device = device)

            memory = None
            frames = [env.render()]
            episode_return = 0.

            for _ in range(max_timesteps):
                action_logits, memory = actor(state, memory = memory)
                action = int(action_logits.argmax(dim = -1).item())

                obs, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)
                frames.append(env.render())

                if terminated or truncated:
                    break

                state = cast_tensor(obs, dtype = torch.float32, device = device)

            if episode_return > best_return:
                best_return = episode_return
                best_frames = frames
                best_seed = int(episode_seed)

    env.close()

    out_dir = Path(video_folder).resolve()
    out_dir.mkdir(parents = True, exist_ok = True)
    mp4_path = out_dir / f'{video_prefix}.mp4'
    gif_path = out_dir / f'{video_prefix}.gif'

    imageio.mimsave(str(mp4_path), best_frames, fps = 50, macro_block_size = 1)
    imageio.mimsave(str(gif_path), best_frames[::2], fps = 25)

    print(f'\n[video] best eval return: {best_return:+.1f} (seed={best_seed}) -> {mp4_path}', flush = True)
    return best_return, str(mp4_path), str(gif_path)

# main

def main(
    env_name = 'LunarLander-v3',
    hierarchies: tuple[int, ...] | int = (1, 4),
    phase_shifts: tuple[int, ...] | None = None,
    phase_schedule = '(20level_0 15level_1)*3',
    num_envs = 2,
    max_steps = 500,
    hidden_dim = 4,
    dim_latent = 4,
    low_rank = 1,
    num_internal_actions = 4,
    population_size = 24,
    noise_scale = 0.15,
    seed = 42,
    cpu = True,
    render = True,
    video_folder = './recordings_hierarchical',
    save_path = './lunar_hierarchical_actor.pt',
    use_wandb = False,
    wandb_project = 'lunar-hierarchical-evo'
):
    accelerator = Accelerator(cpu = cpu)
    is_main = accelerator.is_main_process
    log_wandb = use_wandb and is_main

    if log_wandb:
        wandb.init(project = wandb_project, config = locals())

    hierarchies = resolve_hierarchies(hierarchies)
    assert len(hierarchies) == 2, 'configured for 2 hierarchy levels, e.g. (1, 4)'

    torch.manual_seed(seed)
    np.random.seed(seed)

    actor = HierarchicalActor(
        state_dim = 8,
        hidden_dim = hidden_dim,
        num_actions = 4,
        num_internal_actions = num_internal_actions,
        dim_latent = dim_latent,
        low_rank = low_rank,
        hierarchies = hierarchies,
        phase_shifts = phase_shifts
    )

    env = LunarEnvironment(
        env_name = env_name,
        num_envs = num_envs,
        max_steps = max_steps
    )

    cma = CMAEvoStrategy(
        actor,
        environment = env,
        params_to_optimize = actor.level_parameters(),
        num_generations = 1,
        population_size = population_size,
        noise_scale = noise_scale,
        lora_rank = None,
        sep_cma = True,
        freeze_base = True,
        mirror_sampling = True,
        rollout_fixed_seed = True,
        cpu = cpu,
        accelerator = accelerator,
        verbose = False
    )

    phases = parse_string_schedule(phase_schedule)
    _, current_return, init_ep_rets = env.evaluate(actor)

    if is_main:
        print('\n--- Hierarchical Alternating Evolution (2 Levels) ---')
        print(f'Hierarchies: {hierarchies} | Schedule: {phase_schedule} ({len(phases)} phases)')
        print(f'Population Size: {population_size} | Initial Return: {current_return:+.1f} (max ep: {init_ep_rets.max():+.1f})')
        print('-----------------------------------------------------\n', flush = True)

    start_time = time.time()
    global_gen = 0

    for phase_index, (phase, num_gens) in enumerate(phases, start = 1):
        level_index = 1 if phase == 'level_1' else 0
        if phase in ('level_1', 'all'):
            actor.enable_level_(1)

        start_return = current_return
        global_gen += num_gens

        cma(num_generations = num_gens, scope = phase, verbose = False)
        _, current_return, end_ep_rets = env.evaluate(actor)

        delta_return = current_return - start_return
        max_ep_return = float(end_ep_rets.max())
        landed_pct = 100. * float((end_ep_rets >= 0.).mean())
        elapsed = time.time() - start_time

        phase_label = f'level_{level_index} (dt={hierarchies[level_index]})' if phase != 'all' else 'all'
        if is_main:
            print(
                f'Phase {phase_index:2d} [{phase_label:15s}]: '
                f'start = {start_return:+6.1f} -> end = {current_return:+6.1f} '
                f'(delta = {delta_return:+6.1f}) | max_ep = {max_ep_return:+6.1f} | '
                f'pos_ep = {landed_pct:4.1f}% [{elapsed:.1f}s]',
                flush = True
            )

        if log_wandb:
            wandb.log(dict(
                phase = phase_index,
                scope = phase,
                stride = hierarchies[level_index],
                generation = global_gen,
                start_return = start_return,
                end_return = current_return,
                delta = delta_return,
                max_episode_return = max_ep_return,
                positive_episode_pct = landed_pct,
                avg_steps = env.avg_steps
            ))

    if is_main:
        torch.save(dict(
            actor = actor.state_dict(),
            hierarchies = hierarchies
        ), save_path)

        if render:
            record_best_episode_video(
                actor,
                env_name = env_name,
                video_folder = video_folder,
                max_timesteps = max_steps
            )

    if log_wandb:
        wandb.finish()

    if dist.is_initialized():
        dist.destroy_process_group()
        os._exit(0)

if __name__ == '__main__':
    fire.Fire(main)
