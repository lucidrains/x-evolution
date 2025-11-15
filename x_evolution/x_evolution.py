from __future__ import annotations
from typing import Callable

import torch
from torch import tensor
from torch.nn import Module
import torch.nn.functional as F
from torch.func import functional_call, vmap

from beartype import beartype
from beartype.door import is_bearable

from x_mlps_pytorch.noisable import (
    Noisable,
    with_seed
)

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
        environment: Callable[[Module], float],  # the environment is simply a function that takes in the model and returns a fitness score
        num_generations,
        param_names_to_optimize: list[str] | None = None,
        fitness_to_weighted_factor: Callable[[Tensor], Tensor] = normalize
    ):
        super().__init__()

        self.model = model
        self.environment = environment

        param_names = set(dict(model.named_parameters()).keys())

        # default to all parameters to optimize with evo strategy

        param_names_to_optimize = default(param_names_to_optimize, param_names)

        # validate

        assert all([name in param_names for name in param_names_to_optimize])

        # sort param names and store

        param_names_sorted_list = list(param_names_to_optimize).sort()

        self.param_names_to_optimize = param_names_sorted_list

        # the function that transforms a tensor of fitness floats to the weight for the weighted average of the noise for rolling out 1x1 ES

        self.fitness_to_weighted_factor = fitness_to_weighted_factor

        self.register_buffer('_dummy', tensor(0), persistent = False)

    @property
    def device(self):
        return self._dummy.device

    def evolve_(
        self,
        fitnesses: list[float] | Tensor
    ):
        if isinstance(fitnesses, list):
            fitnesses = tensor(fitnesses)

        fitnesses = fitnesses.to(self.device)

        # they use a simple z-score for the fitnesses, need to figure out the natural ES connection

        noise_weights = self.fitness_to_weighted_factor(fitnesses)
