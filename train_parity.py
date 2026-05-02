# /// script
# dependencies = [
#     "fire",
#     "x-evolution>=0.0.20"
# ]
# ///

import fire
import torch
from torch import nn
import torch.nn.functional as F

from x_evolution import EvoStrategy

# model

class GRUParity(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.emb = nn.Embedding(2, dim)
        self.gru = nn.GRU(dim, dim, batch_first = True)
        self.to_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2))

    def forward(self, x):
        return self.to_out(self.gru(self.emb(x))[0])

# curriculum

class Curriculum:
    def __init__(self, max_length, threshold = 0.01, patience = 50):
        self.seq_len = 1
        self.max_length = max_length
        self.threshold = threshold
        self.patience = patience
        self.criteria = 0

    def step(self, loss):
        self.criteria = (self.criteria + 1) if loss < self.threshold else 0

        if self.criteria >= self.patience:
            self.seq_len = min(self.seq_len + 1, self.max_length)
            self.criteria = 0
            print(f'\n  > incrementing to length {self.seq_len}')

# environment

def parity_environment(model, curriculum, batch_size):
    device = next(model.parameters()).device

    seq = torch.randint(0, 2, (batch_size, curriculum.seq_len), device = device)
    labels = seq.cumsum(dim = -1) % 2

    with torch.no_grad():
        logits = model(seq)
        loss = F.cross_entropy(logits.transpose(-1, -2), labels)

    curriculum.step(loss.item())
    return -loss

# main

def main(
    dim = 64,
    batch_size = 256,
    max_length = 256,
    num_generations = 50_000,
    noise_pop = 100,
):
    model = GRUParity(dim)
    curriculum = Curriculum(max_length = max_length)

    evo = EvoStrategy(
        model,
        environment = lambda model: parity_environment(model, curriculum, batch_size),
        num_generations = num_generations,
        noise_population_size = noise_pop,
        noise_scale = 1e-2,
        noise_scale_clamp_range = (8e-3, 2e-2),
        noise_low_rank = 1,
        learning_rate = 1e-3,
        learned_noise_scale = True,
        noise_scale_learning_rate = 2e-5,
        use_sigma_optimizer = True,
    )

    print(f'Training GRU parity | dim {dim} | pop {noise_pop} | max length {max_length}')

    evo()

if __name__ == '__main__':
    fire.Fire(main)
