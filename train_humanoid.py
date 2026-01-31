# /// script
# dependencies = [
#     "fire",
#     "gymnasium[mujoco]>=1.0.0",
#     "gymnasium[other]",
#     "x-evolution>=0.0.20",
#     "x-mlps-pytorch"
# ]
# ///

# import os
# os.environ["NCCL_P2P_DISABLE"] = "1"
# os.environ["NCCL_IB_DISABLE"] = "1"
# os.environ["MUJOCO_GL"] = "osmesa"

import fire
from shutil import rmtree
import gymnasium as gym
import numpy as np

import torch
from torch.nn import Module, GRU, Linear
import torch.nn.functional as F

# functions

def exists(v):
    return v is not None

def softclamp(t, value):
    return (t / value).tanh() * value

class HumanoidEnvironment(Module):
    def __init__(
        self,
        video_folder = './recordings_humanoid',
        render_every_eps = 100,
        max_steps = 1000,
        repeats = 1,
        vectorized = False,
        num_envs = 1
    ):
        super().__init__()

        self.vectorized = vectorized
        self.num_envs = num_envs

        if vectorized:
            env = gym.make_vec('Humanoid-v5', num_envs = num_envs, render_mode = 'rgb_array')
        else:
            env = gym.make('Humanoid-v5', render_mode = 'rgb_array')

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
            hiddens = None
            last_action = None
            
            dones = torch.zeros(num_envs, device = device, dtype = torch.bool)

            while step < self.max_steps and not dones.all():

                state_torch = torch.from_numpy(state).float().to(device)

                action_logits, hiddens = model(state_torch, hiddens)

                mean, log_var = action_logits.chunk(2, dim = -1)

                # sample and then bound and scale to -0.4 to 0.4

                std = (0.5 * softclamp(log_var, 5.)).exp()
                sampled = mean + torch.randn_like(mean) * std
                action = sampled.tanh() * 0.4

                next_state, reward, truncated, terminated, info = self.env.step(action.detach().cpu().numpy() if self.vectorized else action.item())

                reward_np = np.array(reward) if not isinstance(reward, np.ndarray) else reward
                total_reward_base = torch.from_numpy(reward_np).float().to(device)

                # reward functions

                # encouraged to move forward (1.0) and stay upright (> 1.2 meters)

                z_pos = torch.from_numpy(next_state[..., 0]).float().to(device)
                x_vel = torch.from_numpy(next_state[..., 5]).float().to(device)

                reward_forward = x_vel
                reward_upright = (z_pos > 1.2).float()

                exploration_bonus = std.mean(dim = -1) * 0.05
                penalize_extreme_actions = (mean.abs() > 1.).float().mean(dim = -1) * 0.05

                penalize_action_change = 0.
                if exists(last_action):
                    penalize_action_change = (last_action - action).abs().mean(dim = -1) * 0.1

                total_reward = total_reward_base + reward_forward + reward_upright + exploration_bonus - penalize_extreme_actions - penalize_action_change

                # only add reward if not done

                mask = (~dones).float()
                cum_reward += total_reward * mask

                # update dones

                dones_np = np.array(truncated | terminated) if not isinstance(truncated | terminated, np.ndarray) else (truncated | terminated)
                dones |= torch.from_numpy(dones_np).to(device)

                step += 1

                state = next_state
                last_action = action

        if not self.vectorized:
            return cum_reward.item() / self.repeats

        return cum_reward / self.repeats

# evo strategy

from x_evolution import EvoStrategy

from x_mlps_pytorch.residual_normed_mlp import ResidualNormedMLP

class Model(Module):

    def __init__(self):
        super().__init__()

        self.deep_mlp = ResidualNormedMLP(
            dim_in = 348,
            dim = 256,
            depth = 8,
            residual_every = 2
        )

        self.gru = GRU(256, 256, batch_first = True)

        self.to_pred = Linear(256, 17 * 2, bias = False)

    def forward(self, state, hiddens = None):

        x = self.deep_mlp(state)

        x = x.unsqueeze(-2)
        gru_out, hiddens = self.gru(x, hiddens)
        x = x + gru_out
        x = x.squeeze(-2)

        return self.to_pred(x), hiddens

from torch.optim.lr_scheduler import CosineAnnealingLR

def main(
    vectorized = False,
    num_envs = 8
):
    evo_strat = EvoStrategy(
        Model(),
        environment = HumanoidEnvironment(
            repeats = 1,
            render_every_eps = 200,
            vectorized = vectorized,
            num_envs = num_envs
        ),
        vectorized = vectorized,
        vector_size = num_envs,
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

if __name__ == '__main__':
    fire.Fire(main)
