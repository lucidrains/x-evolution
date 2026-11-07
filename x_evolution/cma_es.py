from __future__ import annotations

from collections.abc import Callable
from functools import reduce
from math import ceil, exp, log, sqrt
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from beartype import beartype
from torch import Tensor, arange, randint, tensor
from torch.nn import Module, ModuleList
from torch_einops_utils import (
    batched_index_select,
    cast_item,
    dict_add,
    safe_cat,
    safe_stack,
    tree_map_tensor_to_device,
)
from tqdm import tqdm

from x_evolution.noisable import quantize_noise_, validate_noise_dtype, with_seed
from x_evolution.x_evolution import (
    MAX_SEED_VALUE,
    EvolutionStrategy,
    ParametersToOptimize,
    default,
    divisible_by,
    exists,
    identity,
    resolve_rewards,
)

# full covariance cma-es over rank r adapters
# rank r on an (m, n) weight -> r * (m + n) dims, covariance (r * (m + n)) ** 2
# rank 1 -> (m + n) ** 2 instead of (m * n) ** 2

class _CMAState:
    def __init__(
        self,
        strategy,
        param_names,
        *,
        freeze_base,
        lora_rank,
        sep_cma,
        mirror_sampling,
        population_size,
        noise_scale,
        noise_dtype,
        sigma_clamp_range
    ):
        self.strategy = strategy
        self.param_names = sorted(param_names)
        self.param_shapes = {name: strategy._named_parameters[name].shape for name in self.param_names}

        self.lora_rank = lora_rank
        self.freeze_base = freeze_base
        self.sep_cma = sep_cma
        self.noise_dtype = noise_dtype
        self.sigma_clamp_range = sigma_clamp_range

        self.dim = 0
        self.param_slices = dict()

        for name in self.param_names:
            m, n = self.param_shapes[name]
            size = lora_rank * (m + n) if exists(lora_rank) else m * n
            self.param_slices[name] = (self.dim, self.dim + size)
            self.dim += size

        device = strategy.cma_device

        self.base_params = {
            name: strategy._named_parameters[name].detach().clone()
            for name in self.param_names
        }

        self.mean = torch.zeros(self.dim, device = device)
        self.C = torch.ones(self.dim, device = device) if sep_cma else torch.eye(self.dim, device = device)
        self.pc = torch.zeros(self.dim, device = device)
        self.ps = torch.zeros(self.dim, device = device)

        self.mirror_sampling = mirror_sampling

        assert population_size >= 2
        self.population_size = population_size
        self.num_parents = population_size // 2

        weights = log(self.num_parents + 0.5) - torch.log(arange(1, self.num_parents + 1, dtype = torch.float32))
        weights /= weights.sum()

        self.weights = weights.to(device)
        self.mu_eff = 1. / (weights ** 2).sum().item()

        dim = self.dim

        self.c_sigma = (self.mu_eff + 2.) / (dim + self.mu_eff + 5.)
        self.d_sigma = 1. + 2. * max(0., sqrt((self.mu_eff - 1.) / (dim + 1.)) - 1.) + self.c_sigma
        self.c_c = (4. + self.mu_eff / dim) / (dim + 4. + 2. * self.mu_eff / dim)
        self.c_1 = 2. / ((dim + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(1. - self.c_1, 2. * (self.mu_eff - 2. + 1. / self.mu_eff) / ((dim + 2.) ** 2 + self.mu_eff))

        if sep_cma:
            self.c_1 = min(1., self.c_1 * (dim + 1.5) / 3.)
            self.c_mu = min(1. - self.c_1, self.c_mu * (dim + 1.5) / 3.)

        self.chi_n = sqrt(dim) * (1. - 1. / (4. * dim) + 1. / (21. * dim ** 2))

        # noise scale is the initial std of the induced weight delta
        # factors with std sigma give a rank r delta with std sigma ** 2

        self.sigma = sqrt(noise_scale) if exists(lora_rank) else noise_scale

        self.generation = 0
        self._update = None

    @property
    def covariance_numel(self):
        return self.dim if self.sep_cma else self.dim ** 2

    @property
    def should_mirror_sample(self):
        if not self.mirror_sampling:
            return False

        if not exists(self.lora_rank):
            return True

        # low rank decode is quadratic around zero, so mirror pairs collapse

        return self.mean.count_nonzero().item() > 0

    def to_device_(self):
        state = dict(mean = self.mean, C = self.C, pc = self.pc, ps = self.ps, weights = self.weights)
        state = tree_map_tensor_to_device(state, self.strategy.cma_device)

        for name, value in state.items():
            setattr(self, name, value)

    @torch.no_grad()
    def decode_(self, vec):
        deltas = dict()

        for name in self.param_names:
            lo, hi = self.param_slices[name]
            param_vec = vec[lo:hi]
            m, n = self.param_shapes[name]

            if exists(self.lora_rank):
                r = self.lora_rank
                b = param_vec[:m * r].reshape(m, r)
                a = param_vec[m * r:].reshape(r, n)
                delta = (b @ a) / sqrt(r)
            else:
                delta = param_vec.reshape(m, n)

            deltas[name] = delta

        return deltas

    @torch.no_grad()
    def set_model_to_(self, vec):
        deltas = self.decode_(vec)

        for name, delta in deltas.items():
            param = self.strategy._named_parameters[name]
            base = self.base_params[name]
            param.data.copy_(base + delta.to(base.device))

    @torch.no_grad()
    def sample_noise_(self, shape, generator):
        noise_dtype = self.noise_dtype

        if exists(noise_dtype) and not noise_dtype.is_floating_point:
            noise = torch.randn(shape, generator = generator)
            noise = quantize_noise_(noise, bits = torch.iinfo(noise_dtype).bits, generator = generator)
        else:
            noise = torch.randn(shape, generator = generator, dtype = default(noise_dtype, torch.float32))

        return noise.to(self.strategy.cma_device)

    @torch.no_grad()
    def sample_(self, num_samples, generator, mirror_sampling = None):
        mirror_sampling = default(mirror_sampling, self.mirror_sampling)

        if mirror_sampling:
            assert (num_samples % 2) == 0
            half = num_samples // 2
            z_half = self.sample_noise_((half, self.dim), generator)
            z = safe_cat((z_half, -z_half), dim = 0)
        else:
            z = self.sample_noise_((num_samples, self.dim), generator)

        if self.sep_cma:
            y = z * self.C.sqrt()
        else:
            # eigen decomposition of C = B diag(d ** 2) B^T

            C = 0.5 * (self.C + self.C.T)
            eigenvalues, eigenvectors = torch.linalg.eigh(C)
            eigenvalues = eigenvalues.clamp(min = 1e-14)
            y = (z * eigenvalues.sqrt()) @ eigenvectors.T

        x = self.mean + self.sigma * y

        return x, z, y

    @torch.no_grad()
    def tell_(self, x, z, y, fitnesses):
        num_parents = self.num_parents
        weights = self.weights

        fitnesses = fitnesses.to(self.strategy.cma_device)

        # rank based parent selection

        selected = fitnesses.topk(num_parents).indices

        y_selected = batched_index_select(y, selected, dim = 0)
        z_selected = batched_index_select(z, selected, dim = 0)

        y_mean = (weights[:, None] * y_selected).sum(dim = 0)
        z_mean = (weights[:, None] * z_selected).sum(dim = 0)

        # mean shift in factor space

        new_mean = self.mean + self.sigma * y_mean

        # sigma path

        ps = (1. - self.c_sigma) * self.ps + sqrt(self.c_sigma * (2. - self.c_sigma) * self.mu_eff) * z_mean
        ps_norm = cast_item(ps.norm())

        self.ps.copy_(ps)

        self.sigma *= exp((self.c_sigma / self.d_sigma) * (ps_norm / self.chi_n - 1.))

        # covariance path

        generation = self.generation + 1
        h_sigma = ps_norm / sqrt(1. - (1. - self.c_sigma) ** (2. * generation)) < (1.4 + 2. / (self.dim + 1.)) * self.chi_n
        h_sigma = float(h_sigma)

        pc = (1. - self.c_c) * self.pc + h_sigma * sqrt(self.c_c * (2. - self.c_c) * self.mu_eff) * y_mean
        self.pc.copy_(pc)

        if self.sep_cma:
            C = (
                (1. - self.c_1 - self.c_mu) * self.C
                + self.c_1 * (pc.square() + (1. - h_sigma) * self.c_c * (2. - self.c_c) * self.C)
                + self.c_mu * (weights[:, None] * y_selected.square()).sum(dim = 0)
            )
            self.C.copy_(C)
        else:
            rank_mu = (weights[:, None] * y_selected).T @ y_selected

            C = (
                (1. - self.c_1 - self.c_mu) * self.C
                + self.c_1 * (torch.outer(pc, pc) + (1. - h_sigma) * self.c_c * (2. - self.c_c) * self.C)
                + self.c_mu * rank_mu
            )
            self.C.copy_(0.5 * (C + C.T))

        self.mean.copy_(new_mean)

        # es-style accumulation uses the weighted mean of the selected perturbations
        # the product of the mean factors can cancel, so do not use decode(mean) here

        if not self.freeze_base:
            deltas = [
                {name: weight * value for name, value in self.decode_(x[index]).items()}
                for weight, index in zip(weights, selected.tolist())
            ]

            self._update = reduce(dict_add, deltas, dict())

        min_sigma, max_sigma = self.sigma_clamp_range
        self.sigma = min(max(self.sigma, min_sigma), max_sigma)

        self.generation = generation

    @torch.no_grad()
    def commit_(self):
        if not exists(self._update):
            return

        for name, delta in self._update.items():
            self.base_params[name].add_(delta.to(self.base_params[name].device))

        self._update = None
        self.mean.zero_()
        self.set_model_to_(self.mean)

    @torch.no_grad()
    def merge_mean_into_base_(self):
        deltas = self.decode_(self.mean)

        for name, delta in deltas.items():
            base = self.base_params[name]
            base.add_(delta.to(base.device))

        self.mean.zero_()
        self.set_model_to_(self.mean)

class CMAEvoStrategy(EvolutionStrategy):

    @beartype
    def __init__(
        self,
        model: Module | list[Module],
        *,
        environment: Callable[[Module], Any] | dict[str, Callable[[Module], Any]],
        num_generations,
        population_size: int | None = None,
        noise_scale = 1e-2,
        noise_dtype: torch.dtype | None = None, # float16 / bfloat16 / int8 - int8 stochastically rounds the sampled noise
        lora_rank: int | None = 1,
        freeze_base = True,
        merge_every: int | None = None,  # fold mean into base every n generations
        mirror_sampling = True,
        sep_cma = False,  # diagonal covariance: O(dim) memory, no eigendecomposition
        params_to_optimize: dict[str, ParametersToOptimize] | ParametersToOptimize | None = None,
        rollout_fixed_seed = True,
        transform_fitness: Callable = identity,
        checkpoint_every = None,
        checkpoint_path = './checkpoints',
        cma_device: str = 'cpu',
        cpu = False,
        verbose = True,
        accelerator: Accelerator | None = None,
        accelerate_kwargs: dict = dict(),
        on_result: Callable | None = None,
        on_generation: Callable | None = None,
        reject_generation_fitnesses_if: Callable[[Tensor], bool] | None = None,
        sync_on_init = True,
        sigma_clamp_range: tuple[float, float] = (1e-12, 1e6)
    ):
        super().__init__()

        self.verbose = verbose
        self.on_result = on_result
        self.on_generation = on_generation
        self.reject_generation_fitnesses_if = reject_generation_fitnesses_if

        if not exists(accelerator):
            accelerator = Accelerator(cpu = cpu, **accelerate_kwargs)

        self.accelerate = accelerator

        self.environment = environment
        self.environments = environment if isinstance(environment, dict) else None

        environments = list(self.environments.values()) if exists(self.environments) else [environment]

        if accelerator.is_main_process:
            for one_environment in environments:
                pre_main_callback = getattr(one_environment, 'pre_main_callback', None)
                if callable(pre_main_callback):
                    self.print('pre_main_callback detected on environment passed in and is invoked')
                    pre_main_callback()

        self.wait_for_everyone()

        # model

        if isinstance(model, list):
            model = ModuleList(model)

        self.model = model

        if sync_on_init:
            self.sync_model_params_and_buffers_()

        self._named_parameters = dict(model.named_parameters())
        self._param_to_name_index = {param: name for name, param in self._named_parameters.items()}

        if exists(lora_rank):
            assert lora_rank >= 1, 'lora_rank must be >= 1 or None'

        # resolve scopes

        self.scopes = None
        self.scope_names = None

        if isinstance(params_to_optimize, dict):
            self.scopes = dict()
            all_names = set()

            for scope_name, scope_params in params_to_optimize.items():
                resolved = self._resolve_params_to_names(scope_params)
                resolved = {name for name in resolved if self._named_parameters[name].ndim == 2}
                assert resolved, f'nothing to optimize in scope {scope_name}'
                self.scopes[scope_name] = resolved
                all_names |= resolved

            self.scope_names = list(self.scopes)
        else:
            if not exists(params_to_optimize):
                all_names = {name for name, param in self._named_parameters.items() if param.ndim == 2}
            else:
                all_names = self._resolve_params_to_names(params_to_optimize)
                all_names = {name for name in all_names if self._named_parameters[name].ndim == 2}

        assert all_names, 'nothing to optimize'

        self.param_names = sorted(all_names)
        self.param_shapes = {name: self._named_parameters[name].shape for name in all_names}

        # a dict of environments gives one reward function per scope

        if exists(self.environments):
            assert exists(self.scopes), 'when environment is a dict, params_to_optimize must also be a dict'
            assert set(self.environments) == set(self.scopes), f'environment keys {list(self.environments)} must match scope names {list(self.scopes)}'

        # dimensions

        def param_dim(name):
            m, n = self.param_shapes[name]
            return lora_rank * (m + n) if exists(lora_rank) else m * n

        total_dim = sum(param_dim(name) for name in all_names)

        if not exists(population_size):
            population_size = 4 + int(3 * log(total_dim))

        if mirror_sampling and (population_size % 2) == 1:
            population_size += 1

        assert population_size >= 2

        self.population_size = population_size
        self.noise_scale = noise_scale
        validate_noise_dtype(noise_dtype)
        self.noise_dtype = noise_dtype
        self.lora_rank = lora_rank
        self.freeze_base = freeze_base
        self.merge_every = merge_every
        self.sep_cma = sep_cma

        if exists(merge_every):
            assert freeze_base, 'merge_every requires freeze_base=True'
            assert merge_every >= 1, 'merge_every must be >= 1'

        assert not (mirror_sampling and not freeze_base and exists(lora_rank)), 'mirror sampling is incompatible with freeze_base=False and a low-rank adapter; pass mirror_sampling=False or freeze_base=True'

        self._mirror_sampling = mirror_sampling
        self.transform_fitness = transform_fitness
        self.sigma_clamp_range = sigma_clamp_range
        self.cma_device = torch.device(cma_device)

        assert 0. < sigma_clamp_range[0] < sigma_clamp_range[1]

        # states are created lazily per parameter subset / scope

        self._states = dict()
        self.default_state = self._get_state(all_names)

        # bookkeeping

        self.num_generations = num_generations
        self.rollout_fixed_seed = rollout_fixed_seed
        self.checkpoint_every = checkpoint_every
        self.checkpoint_folder = Path(checkpoint_path)
        self.checkpoint_folder.mkdir(parents = True, exist_ok = True)

        self.print(f'cma-es | dim: {self.default_state.dim} | covariance: {self.default_state.covariance_numel}')

    @property
    def dim(self):
        return self.default_state.dim

    @property
    def covariance_numel(self):
        return self.default_state.covariance_numel

    @property
    def C(self):
        return self.default_state.C

    @property
    def mean(self):
        return self.default_state.mean

    @property
    def pc(self):
        return self.default_state.pc

    @property
    def ps(self):
        return self.default_state.ps

    @property
    def sigma(self):
        return self.default_state.sigma

    @property
    def mirror_sampling(self):
        return self.default_state.mirror_sampling

    @property
    def weights(self):
        return self.default_state.weights

    @property
    def base_params(self):
        return self.default_state.base_params

    @property
    def generation(self):
        return self.default_state.generation

    def _get_state(self, param_names):
        names = tuple(sorted(name for name in param_names if self._named_parameters[name].ndim == 2))
        assert names, 'nothing to optimize'

        if names not in self._states:
            self._states[names] = _CMAState(
                self,
                names,
                freeze_base = self.freeze_base,
                lora_rank = self.lora_rank,
                sep_cma = self.sep_cma,
                mirror_sampling = self._mirror_sampling,
                population_size = self.population_size,
                noise_scale = self.noise_scale,
                noise_dtype = self.noise_dtype,
                sigma_clamp_range = self.sigma_clamp_range
            )

        return self._states[names]

    def to_cma_device_(self):
        for state in self._states.values():
            state.to_device_()

    # default state convenience methods

    def decode_(self, vec):
        return self.default_state.decode_(vec)

    def set_model_to_(self, vec):
        return self.default_state.set_model_to_(vec)

    def sample_(self, num_samples, generator):
        return self.default_state.sample_(num_samples, generator)

    def tell_(self, x, z, y, fitnesses):
        return self.default_state.tell_(x, z, y, fitnesses)

    def commit_(self):
        return self.default_state.commit_()

    # rewards

    def _evaluate(self, environment, scope):
        rewards = environment(self.model)
        resolved = resolve_rewards(rewards, scope, self.scope_names)

        assert not isinstance(resolved, list), 'per-scope rewards returned, so pass `scope = ...` instead of `params_to_optimize = ...`'

        return resolved

    def _synced_seed(self):
        seed = randint(0, MAX_SEED_VALUE, ()).item()

        if self.is_distributed:
            seed = int(self.accelerate.reduce(tensor(seed, device = self.device), reduction = 'sum').item())

        return seed % (2 ** 31)

    @torch.inference_mode()
    def _run_state(
        self,
        state,
        scope_name,
        environment,
        num_generations,
        filename,
        disable_distributed = False,
        verbose = None
    ):
        verbose = default(verbose, self.verbose)

        if disable_distributed:
            rank, world_size = 0, 1
        else:
            rank = self.accelerate.process_index
            world_size = self.accelerate.num_processes

        is_distributed = world_size > 1

        population = state.population_size

        generation = 1
        fitnesses_across_generations = []
        suffix = f'.{scope_name}' if exists(scope_name) else ''

        while generation <= num_generations:

            # same samples on every rank

            use_mirror = state.should_mirror_sample
            block = world_size * (2 if use_mirror else 1)
            population_round_up = ceil(population / block) * block
            indices = arange(population_round_up).chunk(world_size)[rank]

            generator = torch.Generator()
            generator.manual_seed(self._synced_seed())

            x, z, y = state.sample_(population_round_up, generator, mirror_sampling = use_mirror)

            local_fitnesses = torch.full((len(indices),), -float('inf'))

            maybe_rollout_seed = self._synced_seed() if self.rollout_fixed_seed else None

            pbar = tqdm(indices, total = len(indices), desc = f'Generation {generation}', disable = not (verbose and self.accelerate.is_main_process))

            for local_index, index in enumerate(pbar):
                if index >= population:
                    continue

                state.set_model_to_(x[index])

                fitness = with_seed(maybe_rollout_seed)(self._evaluate)(environment, scope_name)

                local_fitnesses[local_index] = fitness

                if exists(self.on_result):
                    self.on_result(fitness, pbar)

            pbar.close()

            if is_distributed:
                fitnesses = self.accelerate.gather(local_fitnesses.to(self.device)).cpu().reshape(-1)
            else:
                fitnesses = local_fitnesses

            fitnesses = fitnesses[:population]

            if exists(self.transform_fitness):
                fitnesses = self.transform_fitness(fitnesses)

            if exists(self.reject_generation_fitnesses_if) and self.reject_generation_fitnesses_if(fitnesses):
                self.print(f'[{generation}] fitnesses rejected')
                state.set_model_to_(state.mean)
                generation += 1
                continue

            state.tell_(x[:population], z[:population], y[:population], fitnesses)

            if not state.freeze_base:
                state.commit_()
            else:
                state.set_model_to_(state.mean)

                if exists(self.merge_every) and divisible_by(generation, self.merge_every):
                    state.merge_mean_into_base_()

            fitnesses_across_generations.append(fitnesses)

            if exists(self.on_generation) and self.accelerate.is_main_process:
                self.on_generation(generation, fitnesses, self.model)

            self.print(
                f'[{generation}] average fitness: {fitnesses.mean():.3f} | '
                f'best fitness: {fitnesses.max():.3f} | sigma: {state.sigma:.3e}'
            )

            if exists(self.checkpoint_every) and divisible_by(generation, self.checkpoint_every):
                self.checkpoint(f'{filename}{suffix}.{generation}')

            generation += 1

        self.checkpoint(f'{filename}{suffix}.final.{generation}')

        fitnesses = safe_stack(fitnesses_across_generations)
        return default(fitnesses, tensor([], device = self.device))

    @torch.inference_mode()
    def forward(
        self,
        filename = 'evolved.model',
        num_generations = None,
        disable_distributed = False,
        rollback_model_at_end = False,
        verbose = None,
        scope: str | None = None,
        params_to_optimize: ParametersToOptimize | None = None
    ):
        verbose = default(verbose, self.verbose)

        self.model.to(self.device)

        original_state = self.snapshot_model_() if rollback_model_at_end else None

        self.to_cma_device_()

        num_generations = default(num_generations, self.num_generations)

        # pick the parameter subset / scopes to run

        if exists(scope):
            assert exists(self.scopes) and scope in self.scopes, f'scope "{scope}" not found in passed parameter dictionary'
            state = self._get_state(self.scopes[scope])
            environment = self.environments[scope] if exists(self.environments) else self.environment
            targets = [(scope, state, environment)]
        elif exists(params_to_optimize):
            assert not exists(self.environments), 'pass scope = ... when environment is a dict'
            state = self._get_state(self._resolve_params_to_names(params_to_optimize))
            targets = [(None, state, self.environment)]
        elif exists(self.scopes):
            targets = []
            for name in self.scope_names:
                state = self._get_state(self.scopes[name])
                environment = self.environments[name] if exists(self.environments) else self.environment
                targets.append((name, state, environment))
        else:
            targets = [(None, self.default_state, self.environment)]

        all_fitnesses = []

        for scope_name, state, environment in targets:
            fitnesses = self._run_state(
                state,
                scope_name,
                environment,
                num_generations,
                filename,
                disable_distributed = disable_distributed,
                verbose = verbose
            )
            all_fitnesses.append(fitnesses)

        if exists(original_state):
            self.restore_model_(original_state)

        if len(all_fitnesses) == 1:
            return all_fitnesses[0]

        return safe_stack(all_fitnesses, dim = -1)
