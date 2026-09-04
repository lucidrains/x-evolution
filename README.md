## x-evolution

Implementation of various evolutionary algorithms, starting with [evolutionary strategies](https://blog.otoro.net/2017/10/29/visual-evolution-strategies/)

[explained by @bycloud](https://www.youtube.com/watch?v=lLkE9w1NJs0)

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
    environment = lambda model: torch.randint(0, 100, ()), # environment is just a function that takes in the individual model (with unique noise) and outputs a scalar (the fitness) the measure you are selecting for
    noise_population_size = 30, # increase this for better gradient estimates
    noise_scale = 1e-2,         # the scale of the perturbation noise, also the initial noise scale (sigma) if `learned_noise_scale` = True
    num_generations = 100,    # number of generations / training steps
    learning_rate = 1e-3,     # scale on update derived by fitness and perturb noises
    params_to_optimize = None # defaults to all parameters, but can be [str {param name}] or [Parameter]
)

# do evolution with your desired fitness function for so many generations

evo_strat()

# model will be saved under checkpoints/ folder
# can also specify checkpoint_every at init and select the one with your favored fitness score for continued policy gradient learning etc
```

## Distributed

Using the CLI from 🤗

```shell
$ uv run accelerate config
```

Then

```shell
$ uv run accelerate launch train.py
```

Or with `torchrun` (which can do multiple CPU processes for testing)

```shell
$ uv run torchrun --nproc_per_node=<N> train.py --cpu True
```

For a quick demo on LunarLander with 8 processes

```shell
$ uv run torchrun --nproc_per_node=8 train_lunar.py --cpu True --noise_population_size 100 --fitness_to_weighted_factor centered_rank
```

## Sim-to-Real Bipedal Locomotion (Microduck)

You can train a walking gait for the [Microduck](https://github.com/pollen-robotics/reachy2_microduck) biped robot from scratch using `EvoStrategy` and `torchrun`.

With a population size of 128 distributed across 8 CPU processes, a forward walking gait emerges around **150 generations** (~3-4 minutes):

```shell
$ uv run torchrun --nproc_per_node=8 train_microduck.py
```

To render evaluation videos or resume from the saved best checkpoint:

```shell
# render periodic evaluation videos to videos/
$ uv run torchrun --nproc_per_node=8 train_microduck.py --eval_video True --eval_every 50

# resume training from best checkpoint
$ uv run torchrun --nproc_per_node=8 train_microduck.py --resume True
```

Special thanks to [Pollen Robotics 🤗](https://www.pollen-robotics.com/) for their open sourcing of such a polished sim2real env!

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

```bibtex
@misc{sarkar2025evolutionstrategieshyperscale,
    title   = {Evolution Strategies at the Hyperscale},
    author  = {Bidipta Sarkar and Mattie Fellows and Juan Agustin Duque and Alistair Letcher and Antonio León Villares and Anya Sims and Dylan Cope and Jarek Liesen and Lukas Seier and Theo Wolf and Uljad Berdica and Alexander David Goldie and Aaron Courville and Karin Sevegnani and Shimon Whiteson and Jakob Nicolaus Foerster},
    year    = {2025},
    eprint  = {2511.16652},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2511.16652},
}
```

```bibtex
@misc{fortunato2019noisynetworksexploration,
    title   = {Noisy Networks for Exploration},
    author  = {Meire Fortunato and Mohammad Gheshlaghi Azar and Bilal Piot and Jacob Menick and Ian Osband and Alex Graves and Vlad Mnih and Remi Munos and Demis Hassabis and Olivier Pietquin and Charles Blundell and Shane Legg},
    year    = {2019},
    eprint  = {1706.10295},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/1706.10295},
}
```

```bibtex
@article{ha2017visual,
    title   = "A Visual Guide to Evolution Strategies",
    author  = "Ha, David",
    journal = "blog.otoro.net",
    year    = "2017",
    url     = "https://blog.otoro.net/2017/10/29/visual-evolution-strategies/"
}
```

*Nothing makes sense except in the light of evolution* - Theodosius Dobzhansky
