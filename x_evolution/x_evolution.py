from __future__ import annotations
from typing import Callable

import torch
from torch import tensor, is_tensor
from torch.nn import Module
import torch.nn.functional as F
from torch.func import functional_call, vmap

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

def normalize(t, eps = 1e-6):
    return F.layer_norm(t, t.shape[-1:], eps = eps)

# class

class EvoStrategy(Module):

    @beartype
    def __init__(
        self,
        model: Module,
        *,
        environment: Callable[[Module], float],  # the environment is simply a function that takes in the model and returns a fitness score
        num_generations,
        population_size = 30,
        learning_rate = 1e-3, # todo - optimizer
        noise_scale = 1e-3,   # the noise scaling during rollouts with environment, todo - figure out right value and make sure it can also be customized per parameter name through a dict
        param_names_to_optimize: list[str] | None = None,
        fitness_to_weighted_factor: Callable[[Tensor], Tensor] = normalize,
        cpu = False,
        accelerate_kwargs: dict = dict(),
    ):
        super().__init__()

        self.accelerate = Accelerator(cpu = cpu, **accelerate_kwargs)

        self.model = model
        self.noisable_model = Noisable(model)

        self.environment = environment

        param_names = set(dict(model.named_parameters()).keys())

        # default to all parameters to optimize with evo strategy

        param_names_to_optimize = default(param_names_to_optimize, param_names)

        # validate

        assert all([name in param_names for name in param_names_to_optimize])
        assert len(param_names_to_optimize) > 0, 'nothing to optimize'

        # sort param names and store

        param_names_list = list(param_names_to_optimize)
        param_names_list.sort()

        self.param_names_to_optimize = param_names_list

        # hyperparameters

        self.population_size = population_size
        self.num_params = len(param_names_list) # just convenience for generating all the seeds for all the randn for the proposed memory efficient way

        self.num_generations = num_generations

        # the function that transforms a tensor of fitness floats to the weight for the weighted average of the noise for rolling out 1x1 ES

        self.fitness_to_weighted_factor = fitness_to_weighted_factor

        self.noise_scale = noise_scale
        self.learning_rate = learning_rate

        self.register_buffer('_dummy', tensor(0), persistent = False)

    @property
    def device(self):
        return self._dummy.device

    def print(self, *args, **kwargs):
        return self.accelerate.print(*args, **kwargs)

    @torch.inference_mode()
    def evolve_(
        self,
        fitnesses: list[float] | Tensor,
        seeds_for_population: list[int] | Tensor
    ):
        model = self.noisable_model

        if isinstance(fitnesses, list):
            fitnesses = tensor(fitnesses)

        if isinstance(seeds_for_population, list):
            seeds_for_population = tensor(seeds_for_population)

        fitnesses = fitnesses.to(self.device)
        seeds_for_population.to(self.device)

        # they use a simple z-score for the fitnesses, need to figure out the natural ES connection

        noise_weights = self.fitness_to_weighted_factor(fitnesses)

        noise_weights *= self.learning_rate # some learning rate that subsumes another constant

        # update one seed at a time for enabling evolutionary strategy for large models

        for individual_seed, noise_weight in zip(seeds_for_population.tolist(), noise_weights.tolist()):

            individual_param_seeds = with_seed(individual_seed)(torch.randint)(0, MAX_SEED_VALUE, (self.num_params,))

            noise_config = dict(zip(self.param_names_to_optimize, individual_param_seeds.tolist()))

            # set the noise weight

            noise_config = {param_name: (seed, noise_weight) for param_name, seed in noise_config.items()}

            # now update

            model.add_noise_(noise_config)

    @torch.inference_mode()
    def forward(
        self
    ):

        model = self.noisable_model

        for index in range(self.num_generations):
            generation = index + 1

            fitnesses = []

            # predetermine the seeds for each population
            # each seed is then used as a seed for all the parameters

            seeds_for_population = torch.randint(0, MAX_SEED_VALUE, (self.population_size,))

            # now loop through the entire population of noise

            for individual_seed in seeds_for_population.tolist():

                individual_param_seeds = with_seed(individual_seed)(torch.randint)(0, MAX_SEED_VALUE, (self.num_params,))

                noise_config = dict(zip(self.param_names_to_optimize, individual_param_seeds.tolist()))
                noise_config = {param_name: (seed, self.noise_scale) for param_name, seed in noise_config.items()}

                with model.temp_add_noise_(noise_config):
                    fitness = self.environment(model)

                    if is_tensor(fitness):
                        assert fitness.numel() == 1
                        fitness = fitness.item()

                fitnesses.append(fitness)

            # normalize the fitness and weighted sum of all the noise is the update

            fitnesses = tensor(fitnesses).float()

            self.evolve_(fitnesses, seeds_for_population)

            # log

            self.print(f'[{generation}] average fitness: {fitnesses.mean():.3f} | fitness std: {fitnesses.std():.3f}')

        self.print('evolution complete')
