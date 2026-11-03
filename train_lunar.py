# /// script
# dependencies = [
#     "fire",
#     "gymnasium[box2d]>=1.0.0",
#     "gymnasium[other]",
#     "x-evolution>=0.0.20",
#     "x-mlps-pytorch>=0.2.0"
# ]
# ///

import os

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from shutil import rmtree

import fire
import gymnasium as gym
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn import Module
from torch.optim.lr_scheduler import CosineAnnealingLR
from x_mlps_pytorch.residual_normed_mlp import ResidualNormedMLP

from x_evolution import CMAEvoStrategy, EvoStrategy


class LunarEnvironment(Module):
    def __init__(
        self,
        video_folder = './recordings',
        render_every_eps = 500,
        max_steps = 500,
        repeats = 1,
        vectorized = False,
        num_envs = 1
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

    def pre_main_callback(self):
        # the `pre_main_callback` on the environment passed in is called before the start of the evolutionary strategies loop

        rmtree(self.video_folder, ignore_errors = True)

        if not self.vectorized:
            self.env = gym.wrappers.RecordVideo(
                env = self.env,
                video_folder = self.video_folder,
                name_prefix = 'recording',
                episode_trigger = lambda eps_num: (eps_num % self.render_every_eps) == 0,
                disable_logger = True
            )

    def forward(self, model):

        device = next(model.parameters()).device

        seed = torch.randint(0, int(1e6), ())

        num_envs = self.num_envs if self.vectorized else 1
        cum_reward = torch.zeros(num_envs, device = device)

        for _ in range(self.repeats):
            state, _ = self.env.reset(seed = seed.item())

            step = 0
            dones = torch.zeros(num_envs, device = device, dtype = torch.bool)

            while step < self.max_steps and not dones.all():

                state_torch = torch.from_numpy(state).to(device)

                action_logits = model(state_torch)

                action = F.gumbel_softmax(action_logits, hard = True).argmax(dim = -1)

                next_state, reward, truncated, terminated, *_ = self.env.step(action.detach().cpu().numpy() if self.vectorized else action.item())

                reward_np = np.array(reward) if not isinstance(reward, np.ndarray) else reward
                total_reward = torch.from_numpy(reward_np).float().to(device)

                mask = (~dones).float()
                cum_reward += total_reward * mask

                dones_np = np.array(truncated | terminated) if not isinstance(truncated | terminated, np.ndarray) else (truncated | terminated)
                dones |= torch.from_numpy(dones_np).to(device)

                step += 1

                state = next_state

        if not self.vectorized:
            return cum_reward.item() / self.repeats

        return cum_reward / self.repeats

# both algorithms share the same actor, environment, population, noise scale,
# and checkpoint naming. switch with:
#
#   python train_lunar.py --algorithm es
#   python train_lunar.py --algorithm cma

def main(
    algorithm = 'es',                   # es | cma
    vectorized = False,
    num_envs = 8,
    cpu = False,
    fitness_to_weighted_factor = 'normalize',
    noise_population_size = 50,
    noise_scale = 1e-2,
    sep_cma = False,
    num_generations = 50_000,
    env_repeats = 2,
    max_steps = 500,
    checkpoint_every = None,
    checkpoint_path = './checkpoints',
    verbose = True
):
    assert algorithm in ('es', 'cma'), 'algorithm must be es or cma'

    # cma-es does not support vectorized environments

    if algorithm == 'cma' and vectorized:
        print('CMAES: vectorized environment is not supported, using a single environment')
        vectorized = False
        num_envs = 1

    actor = ResidualNormedMLP(dim_in = 8, dim = 24, depth = 2, residual_every = 1, dim_out = 4)

    environment = LunarEnvironment(
        repeats = env_repeats,
        max_steps = max_steps,
        vectorized = vectorized,
        num_envs = num_envs
    )

    def on_result(fitness, pbar):
        pbar.set_postfix(reward = f'{fitness:.3f}')

    if algorithm == 'es':
        evo_strat = EvoStrategy(
            actor,
            environment = environment,
            cpu = cpu,
            fitness_to_weighted_factor = fitness_to_weighted_factor,
            num_generations = num_generations,
            noise_population_size = noise_population_size,
            noise_low_rank = 1,
            noise_scale = noise_scale,
            noise_scale_clamp_range = (5e-3, 2e-2),
            learned_noise_scale = True,
            use_sigma_optimizer = True,
            learning_rate = 1e-3,
            noise_scale_learning_rate = 1e-4,
            use_scheduler = True,
            scheduler_klass = CosineAnnealingLR,
            scheduler_kwargs = dict(T_max = num_generations),
            on_result = on_result,
            checkpoint_every = checkpoint_every,
            checkpoint_path = checkpoint_path,
            verbose = verbose
        )
    else:
        # es evaluates each seed and its mirror, so match rollouts per generation

        evo_strat = CMAEvoStrategy(
            actor,
            environment = environment,
            num_generations = num_generations,
            population_size = noise_population_size * 2,
            mirror_sampling = False,
            sep_cma = sep_cma,
            noise_scale = noise_scale,
            lora_rank = 1,
            freeze_base = False,
            rollout_fixed_seed = True,
            on_result = on_result,
            checkpoint_every = checkpoint_every,
            checkpoint_path = checkpoint_path,
            cpu = cpu,
            verbose = verbose
        )

    evo_strat(filename = algorithm)

    # cleanup

    if dist.is_initialized():
        dist.destroy_process_group()
        os._exit(0)

if __name__ == '__main__':
    fire.Fire(main)
