import pytest
param = pytest.mark.parametrize

import torch
from x_mlps_pytorch import MLP

model = MLP(8, 16, 4)

# each config toggles one feature

@param('kwargs', (
    dict(),
    dict(mirror_sampling = False),
    dict(use_optimizer = False),
    dict(noise_low_rank = 1, params_to_optimize = [model.layers[1].weight]),
    dict(learned_noise_scale = True),
    dict(learned_noise_scale = True, use_sigma_optimizer = False),
    dict(fitness_to_weighted_factor = 'centered_rank'),
    dict(multi_models = True),
))
def test_evo_strat(kwargs):
    from random import randrange
    from x_evolution.x_evolution import EvoStrategy

    kwargs = dict(kwargs)
    multi_models = kwargs.pop('multi_models', False)

    to_optim = [model, MLP(8, 1)] if multi_models else model

    evo_strat = EvoStrategy(
        to_optim,
        environment = lambda model: float(randrange(100)),
        num_generations = 1,
        noise_population_size = 4,
        **kwargs
    )

    evo_strat('evolve', 1)
    evo_strat('more.evolve', 1)
    evo_strat('more.evolve', 1, rollback_model_at_end = True)

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

def test_scoped_parameters():
    from x_evolution.x_evolution import EvoStrategy
    model = MLP(8, 16, 4)

    params_dict = dict(
        first_layer = list(model.layers[0].parameters()),
        last_layer = list(model.layers[-1].parameters())
    )

    evo_strat = EvoStrategy(
        model,
        environment = lambda m: 1.0,
        num_generations = 1,
        params_to_optimize = params_dict,
        noise_population_size = 2,
    )

    evo_strat(scope = 'first_layer', num_generations = 1)

    evo_strat(scope = 'last_layer', num_generations = 1)

    evo_strat(num_generations = 1) # all

    # dynamically pass params_to_optimize in forward

    dynamic_params = list(model.layers[1].parameters())
    evo_strat(params_to_optimize = dynamic_params, num_generations = 1)

# per-scope rewards - w1 -> w2 is the 'main' path, w3 -> w4 is the 'aux' path, each with its own reward

class TwoPathModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.ones(4, 4))
        self.w2 = torch.nn.Parameter(torch.ones(4, 4))
        self.w3 = torch.nn.Parameter(torch.ones(4, 4))
        self.w4 = torch.nn.Parameter(torch.ones(4, 4))

    def forward(self, x):
        return x @ self.w1 @ self.w2, x @ self.w3 @ self.w4

def test_per_scope_rewards():
    from x_evolution.x_evolution import EvoStrategy

    model = TwoPathModel()
    x = torch.ones(1, 4)

    def main_reward(model):
        main_out, _ = model(x.to(model.device))
        return -main_out.square().mean()

    def aux_reward(model):
        _, aux_out = model(x.to(model.device))
        return -aux_out.square().mean()

    scopes = dict(main = [model.w1, model.w2], aux = [model.w3, model.w4])

    def snapshot():
        return {name: param.detach().cpu().clone() for name, param in model.named_parameters()}

    # one environment per scope, optimized phase by phase

    evo_strat = EvoStrategy(
        model,
        environment = dict(main = main_reward, aux = aux_reward),
        params_to_optimize = scopes,
        num_generations = 1,
        noise_population_size = 4
    )

    before = snapshot()

    evo_strat(scope = 'main')
    after_main = snapshot()

    assert not torch.equal(after_main['w1'], before['w1'])
    assert not torch.equal(after_main['w2'], before['w2'])
    assert torch.equal(after_main['w3'], before['w3'])
    assert torch.equal(after_main['w4'], before['w4'])

    evo_strat(scope = 'aux')
    after_aux = snapshot()

    assert not torch.equal(after_aux['w3'], after_main['w3'])
    assert not torch.equal(after_aux['w4'], after_main['w4'])
    assert torch.equal(after_aux['w1'], after_main['w1'])
    assert torch.equal(after_aux['w2'], after_main['w2'])

    # joint population, each scope updated with its own reward - one fitness column per scope

    fitnesses = evo_strat(num_generations = 1)
    assert fitnesses.shape[-1] == 2

    # a single environment returning one reward per scope works the same

    evo_strat = EvoStrategy(
        model,
        environment = lambda model: (main_reward(model), aux_reward(model)),
        params_to_optimize = scopes,
        num_generations = 1,
        noise_population_size = 4
    )

    fitnesses = evo_strat()
    assert fitnesses.shape[-1] == 2

def test_per_scope_rewards_validated():
    from x_evolution.x_evolution import EvoStrategy

    model = TwoPathModel()
    scopes = dict(main = [model.w1, model.w2], aux = [model.w3, model.w4])

    # environment keys must match the scope names

    with pytest.raises(AssertionError):
        EvoStrategy(
            model,
            environment = dict(main = lambda m: 1.),
            params_to_optimize = scopes,
            num_generations = 1
        )

    # exactly one reward per scope, so too few or too many, or dict keys that do not match, are errors

    for environment in (
        lambda m: (1., 2., 3.),
        lambda m: dict(main = 1., other = 2.)
    ):
        evo_strat = EvoStrategy(
            model,
            environment = environment,
            params_to_optimize = scopes,
            num_generations = 1,
            noise_population_size = 2
        )

        with pytest.raises(AssertionError):
            evo_strat(num_generations = 1)
