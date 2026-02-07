# /// script
# dependencies = [
#     "fire",
#     "gymnasium[mujoco]>=1.0.0",
#     "gymnasium-robotics",
#     "moviepy",
#     "tqdm",
#     "x-evolution>=0.1.27",
#     "x-mlps-pytorch"
# ]
# ///

import fire
from shutil import rmtree
import gymnasium as gym
import gymnasium_robotics
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import Module, GRU, Linear
import torch.nn.functional as F

# functions

def exists(v):
    return v is not None

def softclamp(t, value):
    return (t / value).tanh() * value

class AntMazeEnvironment(Module):
    def __init__(
        self,
        env_id = 'AntMaze_UMazeDense-v5',
        video_folder = './recordings_ant_maze',
        render_every_eps = 100,
        max_steps = 1000,
        repeats = 1,
        verbose = False
    ):
        super().__init__()

        self.verbose = verbose

        env = gym.make(env_id, render_mode = 'rgb_array')

        self.env = env
        self.max_steps = max_steps
        self.repeats = repeats
        self.video_folder = video_folder
        self.render_every_eps = render_every_eps

    def pre_main_callback(self):
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

        num_envs = 1
        cum_reward = torch.zeros(num_envs, device = device)

        for _ in range(self.repeats):
            obs, _ = self.env.reset(seed = seed.item())

            step = 0
            hiddens = None
            
            dones = torch.zeros(num_envs, device = device, dtype = torch.bool)

            while step < self.max_steps and not dones.all():

                state = obs['observation']
                goal = obs['desired_goal']

                state_torch = torch.from_numpy(state).float().to(device)
                goal_torch = torch.from_numpy(goal).float().to(device)

                state_goal = torch.cat((state_torch, goal_torch), dim = -1)

                action_logits, hiddens = model(state_goal, hiddens)

                mean, log_var = action_logits.chunk(2, dim = -1)

                std = (0.5 * softclamp(log_var, 5.)).exp()
                sampled = mean + torch.randn_like(mean) * std
                action = sampled.tanh()

                next_obs, reward, truncated, terminated, info = self.env.step(action.detach().cpu().numpy())

                reward_np = np.array(reward) if not isinstance(reward, np.ndarray) else reward
                total_reward_base = torch.from_numpy(reward_np).float().to(device)

                exploration_bonus = std.mean(dim = -1) * 0.01
                penalize_extreme_actions = (mean.abs() > 1.).float().mean(dim = -1) * 0.01

                total_reward = total_reward_base + exploration_bonus - penalize_extreme_actions

                mask = (~dones).float()
                cum_reward += total_reward * mask

                dones_np = np.array(truncated | terminated) if not isinstance(truncated | terminated, np.ndarray) else (truncated | terminated)
                dones |= torch.from_numpy(dones_np).to(device)

                step += 1
                obs = next_obs

        return cum_reward.item() / self.repeats

# model

from x_mlps_pytorch.residual_normed_mlp import ResidualNormedMLP

class Model(Module):

    def __init__(self, state_dim = 27 + 2, action_dim = 8):
        super().__init__()

        self.deep_mlp = ResidualNormedMLP(
            dim_in = state_dim,
            dim = 256,
            depth = 8,
            residual_every = 2
        )

        self.gru = GRU(256, 256, batch_first = True)

        self.to_pred = Linear(256, action_dim * 2, bias = False)

    def forward(self, state, hiddens = None):

        x = self.deep_mlp(state)

        is_batched = x.ndim > 1

        if not is_batched:
            x = x.unsqueeze(0)

        x = x.unsqueeze(-2)
        gru_out, hiddens = self.gru(x, hiddens)
        x = x + gru_out
        x = x.squeeze(-2)

        if not is_batched:
            x = x.squeeze(0)

        return self.to_pred(x), hiddens

# evo strategy

from x_evolution import EvoStrategy
from torch.optim.lr_scheduler import CosineAnnealingLR

def main(
    env_id = 'AntMaze_UMazeDense-v5',
    num_generations = 50_000,
    max_steps = 250,
    noise_population_size = 64,
    verbose = True
):
    # Determine dims from temp env
    temp_env = gym.make(env_id)
    obs_space = temp_env.observation_space
    action_space = temp_env.action_space
    
    state_dim = obs_space['observation'].shape[0] + obs_space['desired_goal'].shape[0]
    action_dim = action_space.shape[0]
    temp_env.close()

    model = Model(state_dim = state_dim, action_dim = action_dim)

    evo_strat = EvoStrategy(
        model,
        environment = AntMazeEnvironment(
            env_id = env_id,
            max_steps = max_steps,
            repeats = 1,
            render_every_eps = 200,
            verbose = verbose
        ),
        num_generations = num_generations,
        noise_population_size = noise_population_size,
        noise_low_rank = 1,
        noise_scale = 1e-2,
        noise_scale_clamp_range = (5e-3, 2e-2),
        learned_noise_scale = True,
        use_sigma_optimizer = True,
        learning_rate = 1e-3,
        noise_scale_learning_rate = 1e-4,
        use_scheduler = True,
        scheduler_klass = CosineAnnealingLR,
        scheduler_kwargs = dict(T_max = num_generations),
        verbose = verbose
    )

    evo_strat()

if __name__ == '__main__':
    fire.Fire(main)
