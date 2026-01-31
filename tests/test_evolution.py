import pytest
param = pytest.mark.parametrize

import torch
from x_mlps_pytorch import MLP

model = MLP(8, 16, 4)

@param('params_to_optimize', (None, [model.layers[1].weight]))
@param('use_optimizer', (False, True))
@param('noise_low_rank', (None, 1))
@param('mirror_sampling', (False, True))
@param('multi_models', (False, True))
@param('learned_sigma', (False, True))
@param('use_sigma_optimizer', (False, True))
def test_evo_strat(
    params_to_optimize,
    use_optimizer,
    noise_low_rank,
    mirror_sampling,
    multi_models,
    learned_sigma,
    use_sigma_optimizer
):
    from random import randrange

    from x_evolution.x_evolution import EvoStrategy

    to_optim = model

    if multi_models:
        to_optim = [model, MLP(8, 1)]

    evo_strat = EvoStrategy(
        to_optim,
        environment = lambda model: float(randrange(100)),
        num_generations = 2,
        learned_noise_scale = learned_sigma,
        params_to_optimize = params_to_optimize,
        use_optimizer = use_optimizer,
        noise_low_rank = noise_low_rank,
        mirror_sampling = mirror_sampling,
        use_sigma_optimizer = use_sigma_optimizer
    )

    evo_strat('evolve', 1)
    evo_strat('more.evolve', 1)

    fitnesses = evo_strat('more.evolve', 2, rollback_model_at_end = True)

@param('vector_size', (2, 4))
def test_evo_strat_vectorized(vector_size):
    from x_evolution.x_evolution import EvoStrategy

    model = MLP(8, 16, 4)

    def environment(model):
        # mock a vectorized environment returning multiple fitness scores
        return torch.randn(vector_size)

    evo_strat = EvoStrategy(
        model,
        environment = environment,
        num_generations = 2,
        vectorized = True,
        vector_size = vector_size,
        noise_population_size = 4
    )

    evo_strat()
