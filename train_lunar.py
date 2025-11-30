# /// script
# dependencies = [
#     "gymnasium[box2d]>=1.0.0",
#     "gymnasium[other]",
#     "x-evolution>=0.0.20"
# ]
# ///

from shutil import rmtree
import gymnasium as gym

import torch
from torch.nn import Module
import torch.nn.functional as F

class LunarEnvironment(Module):
    def __init__(
        self,
        video_folder = './recordings',
        render_every_eps = 500,
        max_steps = 500,
    ):
        super().__init__()

        env = gym.make('LunarLander-v3', render_mode = 'rgb_array')

        rmtree(video_folder, ignore_errors = True)

        env = gym.wrappers.RecordVideo(
            env = env,
            video_folder = video_folder,
            name_prefix = 'recording',
            episode_trigger = lambda eps_num: (eps_num % render_every_eps) == 0,
            disable_logger = True
        )

        self.env = env
        self.max_steps = max_steps

    def forward(self, model):

        device = next(model.parameters()).device

        state, _ = self.env.reset()

        step = 0
        cum_reward = 0.

        while step < self.max_steps:

            state = torch.from_numpy(state).to(device)

            action_logits = model(state)

            action = F.gumbel_softmax(action_logits, hard = True).argmax(dim = -1)

            next_state, reward, truncated, terminated, *_ = self.env.step(action.item())

            cum_reward += float(reward)
            step += 1

            state = next_state

            if truncated or terminated:
                break

        return cum_reward

# evo strategy

from x_evolution import EvoStrategy

from x_mlps_pytorch import MLP

actor = MLP(8, 16, 4)

evo_strat = EvoStrategy(
    actor,
    num_generations = 1000,
    noise_population_size = 50,
    noise_low_rank = 3,
    noise_scale = 1e-1,
    use_optimizer = True,
    environment = LunarEnvironment()
)

evo_strat()
