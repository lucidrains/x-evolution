# /// script
# dependencies = [
#     "fire",
#     "torchvision",
#     "x-mlps-pytorch>=0.2.0",
#     "x-evolution>=0.0.20"
# ]
# ///

import fire
import torch
from torch import nn
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

def mnist_environment(
    model,
    num_envs = 1,
    vectorized = False,
    batch_size = 256
):
    device = next(model.parameters()).device
    
    iters = num_envs if vectorized else 1

    losses = []

    for _ in range(iters):
        dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
        data_iterator = iter(dataloader)
        data, target = next(data_iterator)

        data, target = data.to(device), target.to(device)

        with torch.no_grad():
            logits = model(data.half())
            loss = F.cross_entropy(logits, target)
        
        losses.append(-loss)

    if not vectorized:
        return losses[0]

    return torch.stack(losses)

# evo

from x_evolution import EvoStrategy

def main(
    vectorized = False,
    num_envs = 8,
    batch_size = 256
):
    evo_strat = EvoStrategy(
        model,
        environment = lambda model: mnist_environment(model, num_envs = num_envs, vectorized = vectorized, batch_size = batch_size),
        vectorized = vectorized,
        vector_size = num_envs,
        noise_population_size = 100,
        noise_scale = 1e-2,
        noise_scale_clamp_range = (8e-3, 2e-2),
        noise_low_rank = 1,
        num_generations = 10_000,
        learning_rate = 1e-3,
        learned_noise_scale = True,
        noise_scale_learning_rate = 2e-5
    )

    evo_strat()

if __name__ == '__main__':
    fire.Fire(main)
