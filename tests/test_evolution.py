import pytest
param = pytest.mark.parametrize

import torch
from x_mlps_pytorch import MLP

model = MLP(8, 16, 4)

@param('params_to_optimize', (None, ['layers.1.weight'], [model.layers[1].weight]))
def test_evo_strat(
    params_to_optimize
):
    from random import randrange

    from x_evolution.x_evolution import EvoStrategy

    evo_strat = EvoStrategy(
        model,
        environment = lambda model: float(randrange(100)),
        num_generations = 10,
        params_to_optimize = params_to_optimize
    )

    evo_strat()
