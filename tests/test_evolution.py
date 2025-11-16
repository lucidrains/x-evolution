import pytest

import torch

def test_evo_strat():
    from random import randrange

    from x_evolution.x_evolution import EvoStrategy

    from x_mlps_pytorch import MLP
    model = MLP(8, 16, 4)

    evo_strat = EvoStrategy(
        model,
        environment = lambda model: float(randrange(100)),
        num_generations = 100
    )

    evo_strat()
