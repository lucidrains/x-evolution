## x-evolution (wip)

Implementation of various evolutionary algorithms, starting with evolutionary strategies

## Install

```bash
$ pip install x-evolution
```

## Usage

```python
import torch
from x_evolution import EvoStrategy

# model

from torch import nn
model = torch.nn.Sequential(
    nn.Linear(8, 16),
    nn.ReLU(),
    nn.Linear(16, 4)
)

# evolution wrapper

evo_strat = EvoStrategy(
    model,
    environment = lambda model: torch.randint(0, 100, ()), # environment is just a function that takes in the individual model (with unique noise) and outputs the fitness - you can select for whatever you want here, does not have to be differentiable.
    noise_population_size = 30,
    num_generations = 100,
    learning_rate = 1e-3,
    noise_scale = 1e-3,
    to_optimize = None # can be [str {param name}] or [Parameter]
)

# do evolution with your desired fitness function for so many generations

evo_strat()

# then save your evolved model, maybe for alternating with gradient based training

torch.save(model.state_dict(), './evolved.pt')
```

## Distributed

Using the CLI from 🤗 

```shell
$ accelerate config
```

Then

```shell
$ accelerate launch train.py
```

## Citations

```bibtex
@article{Qiu2025EvolutionSA,
    title   = {Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning},
    author  = {Xin Qiu and Yulu Gan and Conor F. Hayes and Qiyao Liang and Elliot Meyerson and Babak Hodjat and Risto Miikkulainen},
    journal = {ArXiv},
    year    = {2025},
    volume  = {abs/2509.24372},
    url     = {https://api.semanticscholar.org/CorpusID:281674745}
}
```
