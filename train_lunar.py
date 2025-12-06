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
        repeats = 1
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
        self.repeats = repeats

    def forward(self, model):

        device = next(model.parameters()).device

        seed = torch.randint(0, int(1e6), ())

        cum_reward = 0.

        for _ in range(self.repeats):
            state, _ = self.env.reset(seed = seed.item())

            step = 0

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

        return cum_reward / self.repeats

# evo strategy

from x_evolution import EvoStrategy

from x_mlps_pytorch.normed_mlp import MLP

actor = MLP(8, 24, 24, 4)

evo_strat = EvoStrategy(
    actor,
    environment = LunarEnvironment(repeats = 1),
    num_generations = 50_000,
    noise_population_size = 100,
    noise_low_rank = 1,
    noise_scale = 5e-2,
    learned_noise_scale = True,
    use_sigma_optimizer = True,
    learning_rate = 5e-2,
    noise_scale_learning_rate = 1e-2
)

evo_strat()
