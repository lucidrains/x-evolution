# /// script
# dependencies = [
#     "torchvision",
#     "x-evolution>=0.0.20"
# ]
# ///

import torch
from torch import tensor, nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# model

from x_mlps_pytorch.residual_normed_mlp import ResidualNormedMLP

model = nn.Sequential(
    nn.Flatten(),
    ResidualNormedMLP(dim_in = 784, dim = 512, depth = 8, residual_every = 2, dim_out = 10)
).half()

batch_size = 256

# data

dataset = datasets.MNIST('./data', train = True, download = True, transform = transforms.ToTensor())

# fitness as inverse of loss

def loss_mnist(model):
    device = next(model.parameters()).device
    
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
    data_iterator = iter(dataloader)
    data, target = next(data_iterator)

    data, target = data.to(device), target.to(device)

    with torch.no_grad():
        logits = model(data.half())
        loss = F.cross_entropy(logits, target)

    return -loss

# evo

from x_evolution import EvoStrategy

evo_strat = EvoStrategy(
    model,
    environment = loss_mnist,
    noise_population_size = 100,
    noise_scale = 1e-2,
    noise_scale_clamp_range = (5e-3, 2e-2),
    noise_low_rank = 2,
    num_generations = 10_000,
    learning_rate = 1e-3,
    learned_noise_scale = True,
    noise_scale_learning_rate = 1e-5
)

evo_strat()
