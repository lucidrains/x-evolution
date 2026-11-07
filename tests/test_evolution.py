import os

# deterministic accelerator state across tests
os.environ['ACCELERATE_USE_CPU'] = '1'

import pytest

param = pytest.mark.parametrize

import torch
from torch import nn
from x_mlps_pytorch import MLP

from x_evolution import CMAEvoStrategy

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
def test_evo_strat_vectorized_environment(vector_size):
    from x_evolution.x_evolution import EvoStrategy

    model = MLP(8, 16, 4)

    def environment(model):
        # a vectorized environment returns one fitness per env, averaged by the strategy
        return torch.randn(vector_size)

    evo_strat = EvoStrategy(
        model,
        environment = environment,
        num_generations = 2,
        noise_population_size = 4
    )

    evo_strat()

@param('noise_dtype', (torch.float16, torch.bfloat16, torch.int8))
def test_quantized_noise(noise_dtype, tmp_path):
    from x_evolution.x_evolution import EvoStrategy

    torch.manual_seed(0)

    model = nn.Linear(1, 1, bias = False)

    def environment(m):
        return -((m.model.weight - 1.) ** 2).mean().item()

    evo_strat = EvoStrategy(
        model,
        environment = environment,
        num_generations = 30,
        noise_population_size = 16,
        learning_rate = 1e-1,
        noise_scale = 1e-1,
        noise_dtype = noise_dtype,
        fitness_to_weighted_factor = 'centered_rank',
        cpu = True,
        verbose = False,
        checkpoint_path = str(tmp_path)
    )

    initial_loss = ((model.weight - 1.) ** 2).mean().item()
    evo_strat()

    final_loss = ((model.weight - 1.) ** 2).mean().item()

    assert final_loss < initial_loss * 5e-2

    noise = evo_strat.noisable_model.create_noise_((4, 4))

    assert noise.dtype == noise_dtype

    if noise_dtype == torch.int8:
        assert torch.equal(noise, noise.round())

def test_evo_strategy_rollback_is_in_memory(tmp_path):
    from x_evolution.x_evolution import EvoStrategy

    model = nn.Linear(2, 1, bias = False)
    before = {name: value.clone() for name, value in model.state_dict().items()}

    def environment(m):
        return -((m.model.weight - 1.) ** 2).mean().item()

    evo_strat = EvoStrategy(
        model,
        environment = environment,
        num_generations = 1,
        noise_population_size = 2,
        cpu = True,
        verbose = False,
        checkpoint_path = str(tmp_path)
    )

    evo_strat(rollback_model_at_end = True)

    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])

    # rollback stays in memory

    assert not (tmp_path / 'initial.model.pt').exists()

def test_nested_es_meta(tmp_path):
    depth = 3
    torch.manual_seed(0)

    model = nn.Linear(1, 1, bias = False)
    x = torch.ones(1, 1)
    y = torch.ones(1, 1)

    def loss(m):
        with torch.no_grad():
            return ((m(x) - y) ** 2).mean().item()

    records = {level: [] for level in range(depth + 1)}

    def make_level(module, level):
        def on_generation(_, __, m):
            records[level].append(loss(m))

        def environment(m):
            if level == 0:
                return -loss(m)

            inner = make_level(m, level - 1)
            inner(rollback_model_at_end = True)

            # selection sees inner endpoint, rollback keeps innate weights
            return -records[level - 1][-1]

        return CMAEvoStrategy(
            module,
            environment = environment,
            num_generations = 4,
            population_size = 2,
            noise_scale = 1e-1,
            lora_rank = None,
            freeze_base = False,
            mirror_sampling = False,
            on_generation = on_generation,
            cpu = True,
            verbose = False,
            checkpoint_path = str(tmp_path / f'level-{level}')
        )

    initial_loss = loss(model)
    outer = make_level(model, depth)
    outer()
    final_loss = loss(model)

    assert final_loss < initial_loss

    # each level improves over its own run

    for level in range(depth + 1):
        assert records[level]
        assert records[level][-1] < records[level][0]

    # calls per level

    assert len(records[0]) == 2048
    assert len(records[1]) == 256
    assert len(records[2]) == 32
    assert len(records[3]) == 4

def test_evo_strat_rejected_generation_advances():
    from x_evolution.x_evolution import EvoStrategy

    model = MLP(8, 16, 4)
    rejections = []

    def reject(fitnesses):
        rejections.append(1)
        return len(rejections) == 1

    evo_strat = EvoStrategy(
        model,
        environment = lambda model: 1.,
        num_generations = 2,
        noise_population_size = 2,
        reject_generation_fitnesses_if = reject
    )

    fitnesses = evo_strat()

    assert len(rejections) == 2
    assert fitnesses.shape[0] == 1

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


# rank r adapter on an (m, n) weight -> r * (m + n) dims
# covariance is (r * (m + n)) ** 2 instead of (m * n) ** 2

@param('lora_rank, sep_cma, expected_dim, expected_shape', (
    (1, False, 69, (69, 69)),
    (2, False, 138, (138, 138)),
    (None, False, 160, (160, 160)),
    (1, True, 69, (69,)),
))
def test_cma_dimension(lora_rank, sep_cma, expected_dim, expected_shape):
    model = MLP(2, 32, 3)

    cma = CMAEvoStrategy(
        model,
        environment = lambda m: 0.,
        num_generations = 1,
        lora_rank = lora_rank,
        sep_cma = sep_cma,
        cpu = True,
        checkpoint_path = './checkpoints/test-cma-dim'
    )

    assert cma.dim == expected_dim
    assert cma.C.shape == expected_shape
    assert cma.covariance_numel == (expected_dim if sep_cma else expected_dim ** 2)


def test_cma_dimension_subset():
    model = MLP(2, 32, 3)

    cma = CMAEvoStrategy(
        model,
        environment = lambda m: 0.,
        num_generations = 1,
        lora_rank = 1,
        params_to_optimize = [model.layers[0][0].weight],
        cpu = True,
        checkpoint_path = './checkpoints/test-cma-dim'
    )

    assert cma.param_names == ['layers.0.0.weight']
    assert cma.dim == 32 + 2


def test_cma_dimension_wide():
    model = MLP(784, 128, 10)

    cma = CMAEvoStrategy(
        model,
        environment = lambda m: 0.,
        num_generations = 1,
        lora_rank = 1,
        cpu = True,
        checkpoint_path = './checkpoints/test-cma-dim'
    )

    full_dim = (784 * 128) + (128 * 10)
    lora_dim = (784 + 128) + (128 + 10)

    assert cma.dim == lora_dim == 1050
    assert cma.covariance_numel == lora_dim ** 2
    assert full_dim ** 2 > 9000 * cma.covariance_numel


@param('lora_rank, freeze_base, mirror_sampling', (
    (1, True, True),
    (None, False, True),
    (1, False, True),
    (1, False, False),
))
def test_cma_mirror_sampling_hparams(lora_rank, freeze_base, mirror_sampling):
    model = nn.Linear(4, 3, bias = False)

    kwargs = dict(
        environment = lambda m: 0.,
        num_generations = 1,
        population_size = 4,
        lora_rank = lora_rank,
        freeze_base = freeze_base,
        mirror_sampling = mirror_sampling,
        cpu = True,
        verbose = False,
        checkpoint_path = './checkpoints/test-cma-mirror-hparams'
    )

    if lora_rank is not None and not freeze_base and mirror_sampling:
        with pytest.raises(AssertionError):
            CMAEvoStrategy(model, **kwargs)
        return

    cma = CMAEvoStrategy(model, **kwargs)
    assert cma.mirror_sampling is mirror_sampling


def test_cma_merge_every():
    torch.manual_seed(0)

    model = nn.Linear(4, 3, bias = False)
    base = model.weight.detach().clone()

    cma = CMAEvoStrategy(
        model,
        environment = lambda m: -m.weight.square().mean(),
        num_generations = 2,
        population_size = 8,
        noise_scale = 1e-1,
        lora_rank = 1,
        freeze_base = True,
        merge_every = 2,
        mirror_sampling = True,
        cpu = True,
        verbose = False,
        checkpoint_path = './checkpoints/test-cma-merge'
    )

    assert not cma.default_state.should_mirror_sample

    cma()

    assert cma.mean.count_nonzero().item() == 0
    assert not torch.equal(cma.base_params['weight'], base)
    assert torch.equal(model.weight, cma.base_params['weight'])


def test_cma_merge_every_requires_freeze_base():
    model = nn.Linear(4, 3, bias = False)

    with pytest.raises(AssertionError):
        CMAEvoStrategy(
            model,
            environment = lambda m: 0.,
            num_generations = 1,
            lora_rank = 1,
            freeze_base = False,
            merge_every = 2,
            mirror_sampling = False,
            cpu = True,
            verbose = False,
            checkpoint_path = './checkpoints/test-cma-merge'
        )


@param('noise_dtype', (torch.float16, torch.bfloat16, torch.int8))
def test_cma_quantized_noise(noise_dtype, tmp_path):
    torch.manual_seed(0)

    model = nn.Linear(4, 3, bias = False)
    base = model.weight.detach().clone()
    teacher = base + torch.randn(3, 1) @ torch.randn(1, 4)

    def environment(m):
        return -((m.weight - teacher) ** 2).mean()

    cma = CMAEvoStrategy(
        model,
        environment = environment,
        num_generations = 60,
        population_size = 8,
        noise_scale = 1e-1,
        noise_dtype = noise_dtype,
        lora_rank = 1,
        cpu = True,
        verbose = False,
        checkpoint_path = str(tmp_path)
    )

    _, z, _ = cma.sample_(8, torch.Generator().manual_seed(0))

    assert z.dtype == noise_dtype

    if noise_dtype == torch.int8:
        assert torch.equal(z, z.round())

    initial_distance = (model.weight - teacher).norm().item()

    cma()

    final_distance = (model.weight - teacher).norm().item()

    assert final_distance < initial_distance * 5e-2

def test_cma_noise_scale_is_weight_space_std():
    model = nn.Linear(16, 12, bias = False)

    cma = CMAEvoStrategy(
        model,
        environment = lambda m: 0.,
        num_generations = 1,
        population_size = 64,
        noise_scale = 1e-1,
        lora_rank = 1,
        cpu = True,
        verbose = False,
        checkpoint_path = './checkpoints/test-cma-scale'
    )

    generator = torch.Generator().manual_seed(0)
    x, _, _ = cma.sample_(64, generator)

    deltas = torch.stack([cma.decode_(one)['weight'] for one in x])
    std = deltas.std().item()

    assert 0.08 < std < 0.12


@param('sep_cma, num_generations', (
    (False, 100),
    (True, 200),
))
def test_cma_recovers_rank1(sep_cma, num_generations):
    torch.manual_seed(0)

    model = nn.Linear(6, 4, bias = False)
    base = model.weight.detach().clone()

    teacher = base + torch.randn(4, 1) @ torch.randn(1, 6)

    def environment(model):
        return -((model.weight - teacher) ** 2).mean()

    cma = CMAEvoStrategy(
        model,
        environment = environment,
        num_generations = num_generations,
        population_size = 16,
        noise_scale = 1e-1,
        lora_rank = 1,
        freeze_base = True,
        sep_cma = sep_cma,
        cpu = True,
        verbose = False,
        checkpoint_path = './checkpoints/test-cma-rank1'
    )

    initial_distance = (model.weight - teacher).norm().item()

    cma()

    final_distance = (model.weight - teacher).norm().item()

    assert final_distance < initial_distance * 1e-2
    assert -environment(model) < 1e-3


def test_cma_recenter_accumulates_into_base():
    torch.manual_seed(0)

    model = nn.Linear(6, 4, bias = False)
    base = model.weight.detach().clone()
    teacher = base + torch.randn(4, 1) @ torch.randn(1, 6)

    def environment(model):
        return -((model.weight - teacher) ** 2).mean()

    cma = CMAEvoStrategy(
        model,
        environment = environment,
        num_generations = 60,
        population_size = 16,
        noise_scale = 1e-1,
        lora_rank = 1,
        freeze_base = False,
        mirror_sampling = False,
        cpu = True,
        verbose = False,
        checkpoint_path = './checkpoints/test-cma-recenter'
    )

    cma()

    base_moved = (cma.base_params['weight'] - base).norm().item()

    assert base_moved > 1e-2
    assert (model.weight - teacher).norm() < (base - teacher).norm() * 0.5


def test_cma_full_delta():
    torch.manual_seed(0)

    model = nn.Linear(4, 3, bias = False)
    base = model.weight.detach().clone()
    teacher = base + 0.25 * torch.randn(3, 4)

    def environment(model):
        return -((model.weight - teacher) ** 2).mean()

    cma = CMAEvoStrategy(
        model,
        environment = environment,
        num_generations = 80,
        population_size = 16,
        noise_scale = 1e-1,
        lora_rank = None,
        cpu = True,
        verbose = False,
        checkpoint_path = './checkpoints/test-cma-full'
    )

    assert cma.dim == 12

    cma()

    assert -environment(model) < 1e-2


def test_cma_mirror_sampling_and_checkpoint():
    model = nn.Linear(4, 3, bias = False)

    cma = CMAEvoStrategy(
        model,
        environment = lambda m: -m.weight.square().mean(),
        num_generations = 2,
        population_size = 8,
        mirror_sampling = True,
        lora_rank = 1,
        cpu = True,
        verbose = False,
        checkpoint_every = 1,
        checkpoint_path = './checkpoints/test-cma-mirror'
    )

    fitnesses = cma('smoke')

    assert fitnesses.shape == (2, 8)
    assert (cma.population_size % 2) == 0

    # rollback leaves the model untouched

    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    cma('rollback', num_generations = 1, rollback_model_at_end = True)

    for name, param in model.named_parameters():
        assert torch.equal(param, before[name])


def test_cma_ignores_bias_terms():
    model = MLP(2, 32, 3)
    all_params = list(model.parameters())

    cma = CMAEvoStrategy(
        model,
        environment = lambda m: 0.,
        num_generations = 1,
        params_to_optimize = all_params,
        lora_rank = 1,
        cpu = True,
        verbose = False,
        checkpoint_path = './checkpoints/test-cma-bias'
    )

    assert not any(name.endswith('.bias') for name in cma.param_names)
    assert set(cma.param_names) == {
        name for name, param in model.named_parameters() if param.ndim == 2
    }


def test_cma_scopes():
    model = TwoPathModel()
    x = torch.ones(1, 4)

    def main_reward(model):
        device = next(model.parameters()).device
        main_out, _ = model(x.to(device))
        return -main_out.square().mean()

    def aux_reward(model):
        device = next(model.parameters()).device
        _, aux_out = model(x.to(device))
        return -aux_out.square().mean()

    scopes = dict(main = [model.w1, model.w2], aux = [model.w3, model.w4])

    def snapshot():
        return {name: param.detach().cpu().clone() for name, param in model.named_parameters()}

    cma = CMAEvoStrategy(
        model,
        environment = dict(main = main_reward, aux = aux_reward),
        params_to_optimize = scopes,
        num_generations = 1,
        population_size = 4,
        lora_rank = None,
        freeze_base = True,
        cpu = True,
        verbose = False,
        checkpoint_path = './checkpoints/test-cma-scopes'
    )

    before = snapshot()

    cma(scope = 'main')
    after_main = snapshot()

    assert not torch.equal(after_main['w1'], before['w1'])
    assert not torch.equal(after_main['w2'], before['w2'])
    assert torch.equal(after_main['w3'], before['w3'])
    assert torch.equal(after_main['w4'], before['w4'])

    cma(scope = 'aux')
    after_aux = snapshot()

    assert not torch.equal(after_aux['w3'], after_main['w3'])
    assert not torch.equal(after_aux['w4'], after_main['w4'])
    assert torch.equal(after_aux['w1'], after_main['w1'])
    assert torch.equal(after_aux['w2'], after_main['w2'])

    fitnesses = cma(num_generations = 1)
    assert fitnesses.shape[-1] == 2

    # dynamic subset

    cma = CMAEvoStrategy(
        model,
        environment = main_reward,
        num_generations = 1,
        params_to_optimize = [model.w1],
        population_size = 4,
        lora_rank = None,
        freeze_base = True,
        cpu = True,
        verbose = False,
        checkpoint_path = './checkpoints/test-cma-scopes'
    )

    before = snapshot()
    cma(params_to_optimize = [model.w1])
    after = snapshot()

    assert not torch.equal(after['w1'], before['w1'])
    assert torch.equal(after['w2'], before['w2'])

    # single environment returning one reward per scope

    cma = CMAEvoStrategy(
        model,
        environment = lambda m: (main_reward(m), aux_reward(m)),
        params_to_optimize = scopes,
        num_generations = 1,
        population_size = 4,
        lora_rank = None,
        freeze_base = True,
        cpu = True,
        verbose = False,
        checkpoint_path = './checkpoints/test-cma-scopes'
    )

    fitnesses = cma(num_generations = 1)
    assert fitnesses.shape[-1] == 2
