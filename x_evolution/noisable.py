from __future__ import annotations

from contextlib import contextmanager
from functools import partial, wraps

import torch
from torch import Tensor, randn
from torch.func import functional_call
from torch.nn import Module

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

def is_empty(arr):
    return len(arr) == 0

# noise

def randn_low_rank(shape, *, k = None, dtype = None):
    if not exists(k) or len(shape) != 2 or min(shape) <= k:
        return randn(shape, dtype = dtype)

    out, inp = shape

    a = randn((out, k), dtype = dtype)
    b = randn((k, inp), dtype = dtype)

    return (a @ b) * (k ** -0.5)

def validate_noise_dtype(noise_dtype):
    if not exists(noise_dtype):
        return

    assert isinstance(noise_dtype, torch.dtype)
    assert noise_dtype.is_floating_point or noise_dtype in (torch.int8, torch.int16), 'noise_dtype must be a float dtype, torch.int8, or torch.int16'

def quantize_noise_(noise, bits = 8, generator = None):
    # QES eq. 3 - floor + Bernoulli(frac), then clamp to a signed integer lattice

    floor = noise.floor()
    noise = floor + torch.bernoulli(noise - floor, generator = generator)

    bound = (1 << (bits - 1)) - 1
    noise.clamp_(-bound, bound)

    return noise.to(torch.int8 if bits <= 8 else torch.int16)

# temporary seed

def with_seed(seed):
    if not exists(seed):
        return identity

    def decorator(fn):
        @wraps(fn)
        def inner(*args, **kwargs):
            orig_torch_state = torch.get_rng_state()

            has_cuda = torch.cuda.is_available()
            orig_cuda_states = torch.cuda.get_rng_state_all() if has_cuda else None

            torch.manual_seed(seed)

            if has_cuda:
                torch.cuda.manual_seed_all(seed)

            try:
                out = fn(*args, **kwargs)
            finally:
                torch.set_rng_state(orig_torch_state)

                if has_cuda:
                    torch.cuda.set_rng_state_all(orig_cuda_states)

            return out
        return inner

    return decorator

# wrapper

class Noisable(Module):
    def __init__(
        self,
        model: Module,
        noise_scale = 1.,
        overridable_noise_scale = True,
        low_rank = None,
        noise_dtype: torch.dtype | None = None
    ):
        super().__init__()
        assert not is_empty(list(model.parameters()))

        validate_noise_dtype(noise_dtype)

        self.model = model
        self.noise_scale = noise_scale
        self.noise_dtype = noise_dtype

        self.overridable_noise_scale = overridable_noise_scale # if set to True, if a noise scale is given alongside a seed in a tuple (int, float), that value replaces the noise scale rather than scales it again

        # low rank noise (a la lora)
        self.create_noise_fn = randn if not low_rank else partial(randn_low_rank, k = low_rank)

    @property
    def device(self):
        return next(self.model.parameters()).device

    @torch.no_grad()
    def create_noise_(self, shape, dtype = None):
        noise_dtype = default(self.noise_dtype, dtype)

        # int noise - QES stochastic rounding, stored compactly

        if exists(noise_dtype) and not noise_dtype.is_floating_point:
            bits = torch.iinfo(noise_dtype).bits
            noise = self.create_noise_fn(shape, dtype = torch.float32)
            return quantize_noise_(noise, bits = bits)

        return self.create_noise_fn(shape, dtype = noise_dtype)

    @contextmanager
    def temp_add_noise_(
        self,
        noise_for_params = None,
        noise_scale_for_params = None,
        noise_scale = None,
        negate = False
    ):
        self.get_noised_params(noise_for_params, noise_scale_for_params = noise_scale_for_params, noise_scale = noise_scale, inplace = True, negate = negate)

        yield

        self.get_noised_params(noise_for_params, noise_scale_for_params = noise_scale_for_params, noise_scale = noise_scale, inplace = True, negate = not negate)

    def add_noise_(
        self,
        noise_for_params = None,
        noise_scale_for_params = None,
        noise_scale = None,
        add_to_grad = False,
        negate = False
    ):
        self.get_noised_params(noise_for_params, noise_scale_for_params = noise_scale_for_params, noise_scale = noise_scale, inplace = True, add_to_grad = add_to_grad, negate = negate)

    @torch.no_grad()
    def _add_scaled_noise_(self, name, param, noise, scale, inplace, add_to_grad, return_params):
        is_tensor_scale = isinstance(scale, Tensor)

        if inplace and not add_to_grad:
            if is_tensor_scale:
                param.data.add_(noise * scale)
            else:
                param.data.add_(noise, alpha = scale)

        elif inplace and add_to_grad:
            if not exists(param.grad):
                grad = noise * scale if is_tensor_scale else noise.to(param.dtype) * scale
                param.grad = grad.to(param.dtype)
            elif is_tensor_scale:
                param.grad.add_(noise * scale)
            else:
                param.grad.add_(noise, alpha = scale)

        else:
            return_params[name] = param + noise * scale

    def get_noised_params(
        self,
        noise_for_params = None,
        noise_scale_for_params = None,
        inplace = False,
        noise_scale = None,
        negate = False,
        add_to_grad = False
    ):
        noise_for_params = default(noise_for_params, {})
        noise_scale_for_params = default(noise_scale_for_params, {})

        named_params = dict(self.model.named_parameters())
        return_params = named_params if inplace else dict(named_params)

        for name, param in named_params.items():
            noise_or_seed = noise_for_params.get(name, None)

            if not exists(noise_or_seed):
                continue

            param_noise_scale = default(noise_scale, self.noise_scale)

            # determine the noise

            if isinstance(noise_or_seed, int):
                noise = with_seed(noise_or_seed)(self.create_noise_)(param.shape, dtype = param.dtype)

            elif isinstance(noise_or_seed, tuple) and len(noise_or_seed) == 2:
                seed_or_noise, noise_scale_with_seed = noise_or_seed

                # could be seed or tensor

                if isinstance(seed_or_noise, Tensor):
                    noise = seed_or_noise
                else:
                    noise = with_seed(seed_or_noise)(self.create_noise_)(param.shape, dtype = param.dtype)

                # maybe overriding noise scale per param

                if self.overridable_noise_scale:
                    param_noise_scale = noise_scale_with_seed
                else:
                    param_noise_scale = param_noise_scale * noise_scale_with_seed

            elif isinstance(noise_or_seed, Tensor):
                noise = noise_or_seed
            else:
                raise ValueError('invalid type, noise must be float tensor or int')

            # maybe scale noise by tensor

            maybe_noise_scale = noise_scale_for_params.get(name, None)

            if exists(maybe_noise_scale):
                param_noise_scale = param_noise_scale * maybe_noise_scale

            if negate:
                param_noise_scale = -param_noise_scale

            noise = noise.to(self.device)

            self._add_scaled_noise_(name, param, noise, param_noise_scale, inplace, add_to_grad, return_params)

        return return_params

    def forward(
        self,
        *args,
        noise_for_params = None,
        noise_scale_for_params = None,
        noise_scale = None,
        **kwargs
    ):
        if not exists(noise_for_params) or is_empty(noise_for_params):
            return self.model(*args, **kwargs)

        noised_params = self.get_noised_params(noise_for_params, noise_scale_for_params = noise_scale_for_params, noise_scale = noise_scale)

        return functional_call(self.model, noised_params, args, kwargs)
