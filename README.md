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

## CMA-ES

`CMAEvoStrategy` implements full-covariance CMA-ES over a low-rank adapter, so an `(m, n)` weight costs `r * (m + n)` dimensions instead of `m * n`:

```python
from x_evolution import CMAEvoStrategy

cma = CMAEvoStrategy(
    model,
    environment = environment,
    num_generations = 100,
    lora_rank = 1,          # None for the full weight update
    freeze_base = True,     # False recenters the base every generation
    noise_scale = 1e-2      # std of the induced weight delta
)

cma()
```

Pass `sep_cma = True` for diagonal covariance. `params_to_optimize` accepts the same scopes as `EvoStrategy`; pass `scope = 'name'` to update one scope at a time. 1D parameters such as biases are skipped.

Set `merge_every = N` with `freeze_base = True` to fold the current residual into the base every `N` generations. Mirror sampling is skipped automatically on the first generation after a merge.

`train_lunar.py` switches between the two with `--algorithm es` or `--algorithm cma`.

## Distributed

Using the CLI from 🤗

```shell
$ uv run accelerate config
```

Then

```shell
$ uv run accelerate launch train_lunar.py
```

Or with `torchrun` (which can do multiple CPU processes for testing)

```shell
$ uv run torchrun --nproc_per_node=<N> train_lunar.py --cpu True
```

For a quick demo on LunarLander with 8 processes

```shell
$ uv run torchrun --nproc_per_node=8 train_lunar.py --cpu True --noise_population_size 100 --fitness_to_weighted_factor centered_rank
```

## Sim-to-Real Bipedal Locomotion (Microduck)

Train a command-conditioned walking gait for the [Microduck](https://github.com/pollen-robotics/reachy2_microduck) biped robot from scratch using `EvoStrategy` and `torchrun`. The four driven leg joints follow an inline `CPG`; the policy controls swing, knee scale, phase offsets, and the remaining joint targets.

With 8 CPU processes, a stable gait typically emerges within a few hundred generations:

```shell
$ uv run torchrun --nproc_per_node=8 train_microduck.py
```

Training samples random linear and angular velocity commands by default. Evaluations use `--eval_command` (default `0.15,0`); rendered videos show a 3D arrow above the duck's head pointing along the commanded motion. Use `--horizon 800` for 16-second episodes.

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

```bibtex
@misc{dahal2026backprop,
    title  = {Dust: Pretraining Transformers Without Backpropagation},
    author = {Dahal, Samip and Mandal, Bishwas and G{\"u}lbahar, Serdar and Vegesna, Akshay},
    year   = {2026},
    url    = {https://qlabs.sh/research/dust}
}
```

```bibtex
@article{hansen2001completely,
    title   = {Completely Derandomized Self-Adaptation in Evolution Strategies},
    author  = {Hansen, Nikolaus and Ostermeier, Andreas},
    journal = {Evolutionary Computation},
    year    = {2001},
    volume  = {9},
    number  = {2},
    pages   = {159--195},
    doi     = {10.1162/106365601750190398}
}
```

```bibtex
@inproceedings{ros2008simple,
    title     = {A Simple Modification in {CMA-ES} Achieving Linear Time and Space Complexity},
    author    = {Ros, Raymond and Hansen, Nikolaus},
    booktitle = {Parallel Problem Solving from Nature ({PPSN X})},
    series    = {Lecture Notes in Computer Science},
    volume    = {5199},
    pages     = {296--305},
    year      = {2008},
    doi       = {10.1007/978-3-540-87700-4_30}
}
```

```bibtex
@inproceedings{akimoto2014comparison,
    title     = {Comparison-Based Natural Gradient Optimization in High Dimension},
    author    = {Akimoto, Youhei and Auger, Anne and Hansen, Nikolaus},
    booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference ({GECCO})},
    pages     = {373--380},
    year      = {2014},
    doi       = {10.1145/2576768.2598258}
}
```

*Nothing makes sense except in the light of evolution* - Theodosius Dobzhansky
