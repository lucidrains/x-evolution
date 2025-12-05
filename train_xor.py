import torch
from torch import tensor
import torch.nn.functional as F

# model

from torch import nn

model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 2)
)

# fitness as inverse of loss

from x_evolution import EvoStrategy

def loss_xor(model):
    device = next(model.parameters()).device

    data = torch.randint(0, 2, (32, 2))
    labels = data[:, 0] ^ data[:, 1]

    data, labels = tuple(t.to(device) for t in (data, labels))

    with torch.no_grad():
        logits = model(data.float())
        loss = F.cross_entropy(logits, labels)

    return -loss

# evo

evo_strat = EvoStrategy(
    model,
    environment = loss_xor,
    noise_population_size = 100,
    noise_scale = 1e-1,
    noise_low_rank = 1,
    num_generations = 100_000,
    learning_rate = 1e-4
)

evo_strat()
