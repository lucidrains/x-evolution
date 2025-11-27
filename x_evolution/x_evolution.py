from __future__ import annotations
from typing import Callable
from math import ceil
from pathlib import Path

import torch
from torch import tensor, is_tensor, arange, randint
from torch.nn import Module, Parameter
from torch.optim import Adam
import torch.nn.functional as F

from beartype import beartype
from beartype.door import is_bearable

from accelerate import Accelerator

from x_mlps_pytorch.noisable import (
    Noisable,
    with_seed
)

# constants

MAX_SEED_VALUE = int(2 ** 32)

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def normalize(t, eps = 1e-6):
    return F.layer_norm(t, t.shape[-1:], eps = eps)

# class

class EvoStrategy(Module):

    @beartype
    def __init__(
        self,
        model: Module,
        *,
        environment: Callable[[Module], float | int | Tensor],  # the environment is simply a function that takes in the model and returns a fitness score
        num_generations,
        noise_population_size = 30,
        learning_rate = 1e-3,
        noise_scale = 1e-3,                 # the noise scaling during rollouts with environment, todo - figure out right value and make sure it can also be customized per parameter name through a dict
        params_to_optimize: list[str] | Module | list[Module] | list[Parameter] | None = None,
        use_optimizer = False,
        optimizer_klass = Adam,
        optimizer_kwargs: dict = dict(),
        fitness_to_weighted_factor: Callable[[Tensor], Tensor] = normalize,
        checkpoint_every = None,            # saving every number of generations
        checkpoint_path = './checkpoints',
        cpu = False,
        accelerate_kwargs: dict = dict(),
        reject_generation_fitnesses_if: Callable[[Tensor], bool] | None = None
    ):
        super().__init__()

        self.accelerate = Accelerator(cpu = cpu, **accelerate_kwargs)

        self.model = model
        self.noisable_model = Noisable(model)

        self.environment = environment

        named_parameters_dict = dict(model.named_parameters())

        param_to_name_index = {param: name for name, param in named_parameters_dict.items()}

        param_names = set(named_parameters_dict.keys())

        # default to all parameters to optimize with evo strategy

        params_to_optimize = default(params_to_optimize, param_names)

        # if given as list of Parameter, convert to names

        if is_bearable(params_to_optimize, list[Parameter]):
            params_to_optimize = [param_to_name_index[param] for param in set(params_to_optimize)]

        if isinstance(params_to_optimize, Module):
            params_to_optimize = list(params_to_optimize.parameters())

        if is_bearable(params_to_optimize, list[Module]):
            params_to_optimize = list(ModuleList(params_to_optimize).parameters())

        # validate

        assert all([name in param_names for name in params_to_optimize])
        assert len(params_to_optimize) > 0, 'nothing to optimize'

        # sort param names and store

        param_names_list = list(params_to_optimize)
        param_names_list.sort()

        self.param_names_to_optimize = param_names_list

        # hyperparameters

        self.noise_population_size = noise_population_size
        self.num_params = len(param_names_list) # just convenience for generating all the seeds for all the randn for the proposed memory efficient way

        self.num_generations = num_generations

        # the function that transforms a tensor of fitness floats to the weight for the weighted average of the noise for rolling out 1x1 ES

        self.fitness_to_weighted_factor = fitness_to_weighted_factor

        self.noise_scale = noise_scale
        self.learning_rate = learning_rate

        # maybe use optimizer to update, allow for Adam

        self.use_optimizer = use_optimizer

        if use_optimizer:
            optim_params = [named_parameters_dict[name] for name in params_to_optimize]
            self.optimizer = Adam(optim_params, lr = learning_rate, **optimizer_kwargs)

        # rejecting the fitnesses for a certain generation if this function is true

        self.reject_generation_fitnesses_if = reject_generation_fitnesses_if

        # checkpointing

        self.checkpoint_every = checkpoint_every

        self.checkpoint_folder = Path(checkpoint_path)
        self.checkpoint_folder.mkdir(exist_ok = True)

    @property
    def device(self):
        return self.accelerate.device

    def print(self, *args, **kwargs):
        return self.accelerate.print(*args, **kwargs)

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

        # they use a simple z-score for the fitnesses, need to figure out the natural ES connection

        noise_weights = self.fitness_to_weighted_factor(fitnesses)

        if not use_optimizer:
            noise_weights *= self.learning_rate # some learning rate that subsumes another constant

        # update one seed at a time for enabling evolutionary strategy for large models

        for individual_seed, noise_weight in zip(seeds_for_population.tolist(), noise_weights.tolist()):

            individual_param_seeds = with_seed(individual_seed)(torch.randint)(0, MAX_SEED_VALUE, (self.num_params,))

            noise_config = dict(zip(self.param_names_to_optimize, individual_param_seeds.tolist()))

            # set the noise weight

            noise_config = {param_name: (seed, noise_weight) for param_name, seed in noise_config.items()}

            # now update

            model.add_noise_(noise_config, negate = use_optimizer, add_to_grad = use_optimizer)

        if use_optimizer:
            self.optimizer.step()
            self.optimizer.zero_grad()

    def checkpoint(self, filename = 'evolved.model'):

        if self.accelerate.is_main_process:

            filepath = self.checkpoint_folder / f'{filename}.pt'
            torch.save(self.model.state_dict(), str(filepath))

        self.accelerate.wait_for_everyone()

    @torch.inference_mode()
    def forward(
        self,
        filename = 'evolved.model'
    ):

        model = self.noisable_model.to(self.device)

        # get world size, rank, and determine if distributed

        rank = self.accelerate.process_index
        world_size = self.accelerate.num_processes
        is_distributed = world_size > 1

        # prepare the fitnesses tensor, rounded up to the next multiple of the world size for convenience

        pop_size = self.noise_population_size
        num_pop_per_machine = ceil(pop_size / world_size)
        pop_size_round_up = num_pop_per_machine * world_size

        noise_indices = arange(pop_size_round_up).chunk(world_size)[rank]

        # through many generations

        generation = 1

        while generation <= self.num_generations:

            # predetermine the seeds for each population
            # each seed is then used as a seed for all the parameters

            synced_seed = None

            if is_distributed:
                seed_for_pop = randint(0, int(1e9), (), device = self.device)
                synced_seed = self.accelerate.reduce(seed_for_pop, reduction = 'sum').item()

            seeds_for_population = with_seed(synced_seed)(randint)(0, MAX_SEED_VALUE, (pop_size_round_up,))

            # divy up work across machine

            seeds_for_machine = seeds_for_population.chunk(world_size)[rank]

            fitnesses = []

            # now loop through the entire population of noise

            for noise_index, individual_seed in zip(noise_indices, seeds_for_machine):

                if noise_index >= pop_size:
                    fitnesses.append(0)
                    continue

                individual_param_seeds = with_seed(individual_seed)(randint)(0, MAX_SEED_VALUE, (self.num_params,))

                noise_config = dict(zip(self.param_names_to_optimize, individual_param_seeds.tolist()))
                noise_config = {param_name: (seed, self.noise_scale) for param_name, seed in noise_config.items()}

                with model.temp_add_noise_(noise_config):
                    fitness = self.environment(model)

                    if is_tensor(fitness):
                        assert fitness.numel() == 1
                        fitness = fitness.item()

                fitnesses.append(fitness)

            # normalize the fitness and weighted sum of all the noise is the update

            fitnesses = tensor(fitnesses, device = self.device).float()

            # all gather

            if is_distributed:
                fitnesses = self.accelerate.gather(fitnesses)

            # validate fitnesses

            if exists(self.reject_generation_fitnesses_if) and self.reject_generation_fitnesses_if(fitnesses):
                self.print(f'[{generation}] fitnesses rejected')
                continue

            # pass fitnesses to evolve function

            self.evolve_(
                fitnesses[:pop_size],
                seeds_for_population
            )

            # log

            self.print(f'[{generation}] average fitness: {fitnesses.mean():.3f} | fitness std: {fitnesses.std():.3f}')

            # maybe checkpoint

            if (
                exists(self.checkpoint_every) and
                divisible_by(generation, self.checkpoint_every)
            ):
                self.checkpoint(f'{filename}.{generation}.pt')

            # increment generation

            generation += 1

        self.print('evolution complete')

        self.checkpoint(f'{filename}.final.{generation}')

        self.accelerate.end_training()
