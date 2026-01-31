# /// script
# dependencies = [
#     "fire",
#     "gymnasium[box2d]>=1.0.0",
#     "gymnasium[other]",
#     "x-evolution>=0.0.20",
#     "x-mlps-pytorch>=0.2.0"
# ]
# ///

import fire
from shutil import rmtree
import gymnasium as gym
import numpy as np

import torch
from torch.nn import Module
import torch.nn.functional as F
from x_mlps_pytorch.residual_normed_mlp import ResidualNormedMLP
from torch.optim.lr_scheduler import CosineAnnealingLR

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

# evo strategy

from x_evolution import EvoStrategy

def main(
    vectorized = False,
    num_envs = 8
):
    actor = ResidualNormedMLP(dim_in = 8, dim = 24, depth = 2, residual_every = 1, dim_out = 4)

    evo_strat = EvoStrategy(
        actor,
        environment = LunarEnvironment(
            repeats = 2,
            vectorized = vectorized,
            num_envs = num_envs
        ),
        vectorized = vectorized,
        vector_size = num_envs,
        num_generations = 50_000,
        noise_population_size = 50,
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

if __name__ == '__main__':
    fire.Fire(main)
