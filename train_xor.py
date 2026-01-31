import fire
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

# model

from torch import nn

model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 2)
).half()

batch_size = 128

# fitness as inverse of loss

from x_evolution import EvoStrategy

def xor_environment(
    model,
    num_envs = 1,
    vectorized = False,
    batch_size = 128
):
    device = next(model.parameters()).device

    iters = num_envs if vectorized else 1

    losses = []

    for _ in range(iters):
        data = torch.randint(0, 2, (batch_size, 2))
        labels = data[:, 0] ^ data[:, 1]

        data, labels = tuple(t.to(device) for t in (data, labels))

        with torch.no_grad():
            logits = model(data.half())
            loss = F.cross_entropy(logits, labels)
        
        losses.append(-loss)

    if not vectorized:
        return losses[0]

    return torch.stack(losses)

# evo

def main(
    vectorized = False,
    num_envs = 8,
    batch_size = 128
):
    evo_strat = EvoStrategy(
        model,
        environment = lambda model: xor_environment(model, num_envs = num_envs, vectorized = vectorized, batch_size = batch_size),
        vectorized = vectorized,
        vector_size = num_envs,
        noise_population_size = 100,
        noise_low_rank = 1,
        num_generations = 100_000,
        learning_rate = 1e-1,
        noise_scale = 1e-1,
        noise_scale_clamp_range = (0.05, 0.2),
        learned_noise_scale = True,
        noise_scale_learning_rate = 5e-4,
        use_scheduler = True,
        scheduler_klass = LambdaLR,
        scheduler_kwargs = dict(lr_lambda = lambda step: min(1., step / 10.)),
        use_sigma_scheduler = True,
        sigma_scheduler_klass = LambdaLR,
        sigma_scheduler_kwargs = dict(lr_lambda = lambda step: min(1., step / 10.))
    )

    evo_strat()

if __name__ == '__main__':
    fire.Fire(main)
