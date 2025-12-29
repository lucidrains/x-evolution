# /// script
# dependencies = [
#     "gymnasium[mujoco]>=1.0.0",
#     "gymnasium[other]",
#     "x-evolution>=0.0.20",
#     "x-mlps-pytorch"
# ]
# ///

import os
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["MUJOCO_GL"] = "osmesa"

from shutil import rmtree
import gymnasium as gym
import numpy as np

import torch
from torch.nn import Module
import torch.nn.functional as F

def softclamp(t, value):
    return (t / value).tanh() * value

class HumanoidEnvironment(Module):
    def __init__(
        self,
        video_folder = './recordings_humanoid',
        render_every_eps = 100,
        max_steps = 1000,
        repeats = 1
    ):
        super().__init__()

        # Humanoid-v5
        env = gym.make('Humanoid-v5', render_mode = 'rgb_array')

        self.env = env
        self.max_steps = max_steps
        self.repeats = repeats
        self.video_folder = video_folder
        self.render_every_eps = render_every_eps

    def pre_main_callback(self):
        # the `pre_main_callback` on the environment passed in is called before the start of the evolutionary strategies loop

        rmtree(self.video_folder, ignore_errors = True)

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

        cum_reward = 0.

        for _ in range(self.repeats):
            state, _ = self.env.reset(seed = seed.item())

            step = 0

            while step < self.max_steps:

                state = torch.from_numpy(state).float().to(device)

                action_logits = model(state)

                mean, log_var = action_logits.chunk(2, dim = -1)

                # sample and then bound and scale to -0.4 to 0.4

                std = softclamp((0.5 * log_var).exp(), 10.)
                sampled = mean + torch.randn_like(mean) * std
                action = sampled.tanh() * 0.4

                next_state, reward, truncated, terminated, *_ = self.env.step(action.detach().cpu().numpy())

                cum_reward += float(reward)
                step += 1

                state = next_state

                if truncated or terminated:
                    break

        return cum_reward / self.repeats

# evo strategy

from x_evolution import EvoStrategy

from x_mlps_pytorch.residual_normed_mlp import ResidualNormedMLP

actor = ResidualNormedMLP(
    dim_in = 348, # state
    dim = 256,
    depth = 8,
    residual_every = 2,
    dim_out = 17 * 2 # action mean logvar
)

from torch.optim.lr_scheduler import CosineAnnealingLR

evo_strat = EvoStrategy(
    actor,
    environment = HumanoidEnvironment(repeats = 2),
    num_generations = 50_000,
    noise_population_size = 200,
    noise_low_rank = 1,
    noise_scale = 1e-2,
    noise_scale_clamp_range = (5e-3, 2e-2),
    learned_noise_scale = True,
    use_sigma_optimizer = True,
    learning_rate = 1e-3,
    noise_scale_learning_rate = 1e-4,
    use_scheduler = True,
    scheduler_klass = CosineAnnealingLR,
    scheduler_kwargs = dict(T_max = 50_000)
)

evo_strat()
