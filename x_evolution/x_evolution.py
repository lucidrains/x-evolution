from __future__ import annotations
from typing import Callable

from math import ceil
from pathlib import Path
from functools import partial

import torch
from torch import tensor, Tensor, stack, is_tensor, arange, randint
from torch.nn import Module, ModuleList, Parameter, ParameterList
from torch.optim import SGD, Adam, Optimizer
from torch.optim.lr_scheduler import LRScheduler

import torch.nn.functional as F

from beartype import beartype
from beartype.door import is_bearable

from accelerate import Accelerator

from x_mlps_pytorch.noisable import (
    Noisable,
    with_seed
)

from einops import pack

# constants

MAX_SEED_VALUE = int(2 ** 32)

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

def divisible_by(num, den):
    return (num % den) == 0

def normalize(t, eps = 1e-6):
    return F.layer_norm(t, t.shape, eps = eps)

def accum_grad_(t, value):
    if not exists(t.grad):
        t.grad = value.clone()
    else:
        t.grad.add_(value)

# class

class EvoStrategy(Module):

    @beartype
    def __init__(
        self,
        model: Module | list[Module],
        *,
        environment: Callable[[Module], float | int | Tensor],  # the environment is simply a function that takes in the model and returns a fitness score
        num_generations,
        noise_population_size = 30,
        learning_rate = 1e-3,
        mirror_sampling = True,
        params_to_optimize: list[str] | Module | list[Module] | list[Parameter] | None = None,
        noise_low_rank: int | None = None,
        rollout_fixed_seed = True,
        noise_scale = 1e-2,  # the noise scaling during rollouts with environment, todo - figure out right value and make sure it can also be customized per parameter name through a dict
        learned_noise_scale = False,
        noise_scale_learning_rate = 1e-5,
        noise_scale_clamp_range: tuple[float, float] = (1e-3, 1e-1),
        use_optimizer = True,
        optimizer_klass: type[Optimizer] | Callable = partial(SGD, nesterov = True, momentum = 0.1, weight_decay = 1e-2),
        optimizer_kwargs: dict = dict(),
        use_sigma_optimizer = True,
        sigma_optimizer_klass: type[Optimizer] | Callable = partial(SGD, nesterov = True, momentum = 0.1),
        sigma_optimizer_kwargs: dict = dict(),
        use_scheduler = False,
        scheduler_klass: type[LRScheduler] | None = None,
        scheduler_kwargs: dict = dict(),
        use_sigma_scheduler = False,
        sigma_scheduler_klass: type[LRScheduler] | None = None,
        sigma_scheduler_kwargs: dict = dict(),
        transform_fitness: Callable = identity,
        fitness_to_weighted_factor: Callable[[Tensor], Tensor] = normalize,
        checkpoint_every = None,            # saving every number of generations
        checkpoint_path = './checkpoints',
        cpu = False,
        verbose = True,
        accelerator: Accelerator | None = None,
        accelerate_kwargs: dict = dict(),
        reject_generation_fitnesses_if: Callable[[Tensor], bool] | None = None
    ):
        super().__init__()
        self.verbose = verbose

        if not exists(accelerator):
            accelerator = Accelerator(cpu = cpu, **accelerate_kwargs)

        self.accelerate = accelerator

        # environment - with optional init

        self.environment = environment

        if accelerator.is_main_process:
            if hasattr(environment, 'pre_main_callback') and callable(environment.pre_main_callback):
                self.print('pre_main_callback detected on environment passed in and is invoked')
                environment.pre_main_callback()

        accelerator.wait_for_everyone()

        # take care of model and parameters

        if isinstance(model, list):
            model = ModuleList(model)

        self.model = model
        self.noisable_model = Noisable(model, low_rank = noise_low_rank)

        # use prepare and run through environment once to sync params

        wrapped_model = accelerator.prepare(model)

        with torch.no_grad():
            environment(wrapped_model)

        # get param dictionary

        named_parameters_dict = dict(model.named_parameters())

        param_to_name_index = {param: name for name, param in named_parameters_dict.items()}

        param_names = set(named_parameters_dict.keys())

        # default to all parameters to optimize with evo strategy

        params_to_optimize = default(params_to_optimize, param_names)

        # if Modules given, convert to Parameters
        # then convert Parameters to names

        if isinstance(params_to_optimize, Module):
            params_to_optimize = list(params_to_optimize.parameters())

        if is_bearable(params_to_optimize, list[Module]):
            params_to_optimize = list(ModuleList(params_to_optimize).parameters())

        if is_bearable(params_to_optimize, list[Parameter]):
            params_to_optimize = [param_to_name_index[param] for param in set(params_to_optimize)]

        # validate

        assert all([name in param_names for name in params_to_optimize])
        assert len(params_to_optimize) > 0, 'nothing to optimize'

        # sort param names and store

        param_names_list = list(params_to_optimize)
        param_names_list.sort()

        self.param_names_to_optimize = param_names_list

        # shapes

        self.param_shapes = {name: param.shape for name, param in named_parameters_dict.items()}

        # dtypes

        self.param_dtypes = {name: param.dtype for name, param in named_parameters_dict.items()}
 
        # hyperparameters

        self.noise_population_size = noise_population_size
        self.num_params = len(param_names_list) # just convenience for generating all the seeds for all the randn for the proposed memory efficient way

        self.num_generations = num_generations

        # the function that transforms a tensor of fitness floats to the weight for the weighted average of the noise for rolling out 1x1 ES

        self.transform_fitness = transform_fitness # a function that gets called before converting to weights for the weighted noise update - eventually get rank normalization

        self.fitness_to_weighted_factor = fitness_to_weighted_factor

        self.mirror_sampling = mirror_sampling # mirror / antithetical sampling - reducing variance by doing positive + negative of noise and subtracting

        self.learning_rate = learning_rate

        # noise scale, which can be fixed or learned

        self.noise_scale = noise_scale

        self.learned_noise_scale = learned_noise_scale

        if learned_noise_scale:
            self.sigmas = ParameterList([Parameter(torch.ones_like(param) * noise_scale) for _, param in named_parameters_dict.items()])
            self.param_name_to_sigma_index = {name: i for i, name in enumerate(named_parameters_dict.keys())}

            min_noise_scale, max_noise_scale = noise_scale_clamp_range
            assert min_noise_scale > 0. and min_noise_scale < max_noise_scale

            self.sigma_clamp_ = lambda t: t.clamp_(*noise_scale_clamp_range)

        self.noise_scale_learning_rate = noise_scale_learning_rate

        # rolling out with a fixed seed

        self.rollout_fixed_seed = rollout_fixed_seed

        # maybe use optimizer to update, allow for Adam

        self.use_optimizer = use_optimizer

        if use_optimizer:
            optim_params = [named_parameters_dict[name] for name in params_to_optimize]
            self.optimizer = optimizer_klass(optim_params, lr = learning_rate, **optimizer_kwargs)

        self.use_sigma_optimizer = use_sigma_optimizer

        if learned_noise_scale and use_sigma_optimizer:
            self.sigma_optimizer = sigma_optimizer_klass(self.sigmas, lr = noise_scale_learning_rate, **sigma_optimizer_kwargs)

        # rejecting the fitnesses for a certain generation if this function is true

        self.use_scheduler = use_scheduler

        if use_scheduler and exists(scheduler_klass) and use_optimizer:
            self.scheduler = scheduler_klass(self.optimizer, **scheduler_kwargs)

        self.use_sigma_scheduler = use_sigma_scheduler

        if use_sigma_scheduler and exists(sigma_scheduler_klass) and use_sigma_optimizer:
            self.sigma_scheduler = sigma_scheduler_klass(self.sigma_optimizer, **sigma_scheduler_kwargs)

        self.reject_generation_fitnesses_if = reject_generation_fitnesses_if

        # checkpointing

        self.checkpoint_every = checkpoint_every

        self.checkpoint_folder = Path(checkpoint_path)
        self.checkpoint_folder.mkdir(exist_ok = True)

    @property
    def device(self):
        return self.accelerate.device

    def print(self, *args, **kwargs):
        if not self.verbose:
            return

        return self.accelerate.print(*args, **kwargs)

    def _get_noise_scale(self, param_name):
        if not self.learned_noise_scale:
            return self.noise_scale

        sigma_index = self.param_name_to_sigma_index[param_name]
        return self.sigmas[sigma_index]

    @torch.inference_mode()
    def evolve_(
        self,
        fitnesses: list[float] | Tensor,
        seeds_for_population: list[int] | Tensor
    ):
        use_optimizer = self.use_optimizer
        model = self.noisable_model

        if isinstance(fitnesses, list):
            fitnesses = tensor(fitnesses)

        if isinstance(seeds_for_population, list):
            seeds_for_population = tensor(seeds_for_population)

        fitnesses = fitnesses.to(self.device)
        seeds_for_population = seeds_for_population.to(self.device)

        # maybe transform fitnesses

        if exists(self.transform_fitness):
            fitnesses = self.transform_fitness(fitnesses)

        # maybe normalize the fitness with z-score

        fitnesses = self.fitness_to_weighted_factor(fitnesses)

        if self.mirror_sampling:
            fitness_pos, fitness_neg = fitnesses.unbind(dim = -1)
            weights = fitness_pos - fitness_neg
        else:
            weights = fitnesses

        weights /= self.noise_population_size * (2. if self.mirror_sampling else 1.)

        # update one seed at a time for enabling evolutionary strategy for large models

        for individual_seed, weight in zip(seeds_for_population.tolist(), weights.tolist()):

            individual_param_seeds = with_seed(individual_seed)(torch.randint)(0, MAX_SEED_VALUE, (self.num_params,))

            # setup noise config

            noise_config = dict(zip(self.param_names_to_optimize, individual_param_seeds.tolist()))

            noise_config_with_weight = dict()

            # loop through each noise

            for param_name, seed in noise_config.items():

                shape = self.param_shapes[param_name]
                dtype = self.param_dtypes[param_name]

                # reproduce the noise from the seed

                noise = with_seed(seed)(model.create_noise_fn)(shape, dtype = dtype)
                noise = noise.to(self.device)

                # scale the weight

                noise_scale = self._get_noise_scale(param_name)

                scaled_weight = weight / noise_scale

                if not use_optimizer:
                    scaled_weight = scaled_weight * self.learning_rate

                # set the noise weight

                noise_config_with_weight[param_name] = (noise, scaled_weight)

                # maybe learned sigma

                if self.learned_noise_scale:
                    one_grad_sigma = weight * (noise ** 2 - 1) / noise_scale

                    sigma_index = self.param_name_to_sigma_index[param_name]
                    sigma = self.sigmas[sigma_index]

                    if self.use_sigma_optimizer:
                        accum_grad_(sigma, -one_grad_sigma)
                    else:
                        sigma.add_(one_grad_sigma * self.noise_scale_learning_rate)

            # now update params for one seed

            model.add_noise_(noise_config_with_weight, negate = use_optimizer, add_to_grad = use_optimizer)

        if use_optimizer:
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.use_scheduler and exists(self.scheduler):
                self.scheduler.step()

        if self.learned_noise_scale:
            if self.use_sigma_optimizer:
                self.sigma_optimizer.step()
                self.sigma_optimizer.zero_grad()

                if self.use_sigma_scheduler and exists(self.sigma_scheduler):
                    self.sigma_scheduler.step()

            for sigma in self.sigmas:
                self.sigma_clamp_(sigma)

    def checkpoint(self, filename = 'evolved.model'):

        if self.accelerate.is_main_process:

            filepath = self.checkpoint_folder / f'{filename}.pt'
            torch.save(self.model.state_dict(), str(filepath))

        self.accelerate.wait_for_everyone()

    @torch.inference_mode()
    def forward(
        self,
        filename = 'evolved.model',
        num_generations = None,
        disable_distributed = False,
        rollback_model_at_end = False,
        verbose = None
    ):
        verbose = default(verbose, self.verbose)

        model = self.noisable_model.to(self.device)

        # maybe save model for rolling back (for meta-evo)

        if rollback_model_at_end:
            self.checkpoint('initial.model')

        # maybe sigmas

        if self.learned_noise_scale:
             self.sigmas = self.sigmas.to(self.device)

        # get world size, rank, and determine if distributed

        if disable_distributed:
            rank, world_size = 0, 1
        else:
            rank = self.accelerate.process_index
            world_size = self.accelerate.num_processes

        is_distributed = world_size > 1

        # prepare the fitnesses tensor, rounded up to the next multiple of the world size for convenience

        pop_size = self.noise_population_size
        num_pop_per_machine = ceil(pop_size / world_size)
        pop_size_round_up = num_pop_per_machine * world_size

        noise_indices = arange(pop_size_round_up).chunk(world_size)[rank]

        # maybe synced seed

        def maybe_get_synced_seed():
            seed = randint(0, int(1e9), (), device = self.device)

            if is_distributed:
                seed = self.accelerate.reduce(seed, reduction = 'sum')

            return seed.item()

        # through many generations

        num_generations = default(num_generations, self.num_generations)

        generation = 1

        fitnesses_across_generations = []

        # loop through generations

        while generation <= num_generations:

            # predetermine the seeds for each population
            # each seed is then used as a seed for all the parameters

            seeds_for_population = with_seed(maybe_get_synced_seed())(randint)(0, MAX_SEED_VALUE, (pop_size_round_up,))

            # divy up work across machine

            seeds_for_machine = seeds_for_population.chunk(world_size)[rank]

            fitnesses = []

            # function for fitness

            def rollout_for_fitness():
                fitness = self.environment(model)

                if is_tensor(fitness):
                    assert fitness.numel() == 1
                    fitness = fitness.item()

                return fitness

            # seeds

            maybe_rollout_seed = maybe_get_synced_seed() if self.rollout_fixed_seed else None

            # now loop through the entire population of noise

            for noise_index, individual_seed in zip(noise_indices, seeds_for_machine):

                if noise_index >= pop_size:
                    fitnesses.append([0., 0.] if self.mirror_sampling else 0.)
                    continue

                individual_param_seeds = with_seed(individual_seed)(randint)(0, MAX_SEED_VALUE, (self.num_params,))

                noise_config = dict(zip(self.param_names_to_optimize, individual_param_seeds.tolist()))

                # determine noise scale, which can be fixed or learned

                noise_config_with_scale = dict()

                for param_name, seed in noise_config.items():

                    noise_scale = self._get_noise_scale(param_name)

                    noise_config_with_scale[param_name] = (seed, noise_scale)

                # maybe roll out with a fixed seed

                with model.temp_add_noise_(noise_config_with_scale):
                    fitness = with_seed(maybe_rollout_seed)(rollout_for_fitness)()

                if not self.mirror_sampling:
                    fitnesses.append(fitness)
                    continue

                # handle mirror sampling

                with model.temp_add_noise_(noise_config_with_scale, negate = True):
                    fitness_mirrored = with_seed(maybe_rollout_seed)(rollout_for_fitness)()

                fitnesses.append([fitness, fitness_mirrored])

            # normalize the fitness and weighted sum of all the noise is the update

            fitnesses = tensor(fitnesses, device = self.device).float()

            # all gather

            if is_distributed:
                fitnesses = self.accelerate.gather(fitnesses)

            # remove padding

            fitnesses = fitnesses[:pop_size]
            seeds_for_population = seeds_for_population[:pop_size]

            # store

            fitnesses_across_generations.append(fitnesses)

            # validate fitnesses

            if exists(self.reject_generation_fitnesses_if) and self.reject_generation_fitnesses_if(fitnesses):
                self.print(f'[{generation}] fitnesses rejected')
                continue

            # pass fitnesses to evolve function

            self.evolve_(fitnesses, seeds_for_population)

            # log

            msg = f'[{generation}] average fitness: {fitnesses.mean():.3f} | fitness std: {fitnesses.std():.3f}'

            if self.learned_noise_scale:
                packed_sigma, _ = pack(list(self.sigmas), '*')
                avg_sigma = packed_sigma.mean().item()
                msg += f' | average sigma: {avg_sigma:.3f}'

            self.print(msg)

            # maybe checkpoint

            if (
                exists(self.checkpoint_every) and
                divisible_by(generation, self.checkpoint_every)
            ):
                self.checkpoint(f'{filename}.{generation}.pt')

            # increment generation

            generation += 1

        self.print('evolution complete')

        # final checkpoint

        self.checkpoint(f'{filename}.final.{generation}')

        # maybe rollback

        if rollback_model_at_end:
            orig_state_dict = torch.load(str(self.checkpoint_folder / 'initial.model.pt'), weights_only = True)

            self.model.load_state_dict(orig_state_dict)

        # return fitnesses across generations
        # for meta-evolutionary (nesting EvoStrategy within the environment of another and optimizing some meta-network)

        return stack(fitnesses_across_generations)
