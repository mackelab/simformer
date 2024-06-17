
# All-in-one simulation-based inference

Amortized Bayesian inference trains neural networks to solve stochastic inference problems using model simulations, thereby making it possible to rapidly perform Bayesian inference for any newly observed data. However, current simulation-based amortized inference methods are simulation-hungry and inflexible: They require the specification of a fixed parametric prior, simulator, and inference tasks ahead of time. Here, we present a new amortized inference method -- the Simformer -- which overcomes these limitations. By training a probabilistic diffusion model with transformer architectures, the Simformer outperforms current state-of-the-art amortized inference approaches on benchmark tasks and is substantially more flexible: It can be applied to models with function-valued parameters, it can handle inference scenarios with missing or unstructured data, and it can sample arbitrary conditionals of the joint distribution of parameters and data, including both posterior and likelihood. We showcase the performance and flexibility of the Simformer on simulators from ecology, epidemiology, and neuroscience, and demonstrate that it opens up new possibilities and application domains for amortized Bayesian inference on simulation-based models.


## Installation

If you have conda installed, you should first load a new environment. A minimal environment with
recommended cuda version for JAX is provided in `src/environment.yml`.

```bash
conda env create --file=src/environment.yml
conda activate simformer
pip install -e src/probjax[cuda]
pip install -e src/scoresbibm
```

We recommend installing it on a CUDA capable machine, as the experiments heavily benefit
from GPU acceleration. The above will install the CUDA version of JAX. If you do not have 
a CUDA capable machine, you can install the CPU version by dropping the `[cuda]` flag. 


## Reproducing the experiments
### Running the experiments

We use [Hydra](https://github.com/facebookresearch/hydra) to manage the configurations.  See `src/scoresbibm/config` for all configurations and defaults.

A new command-line script `scoresbi` is installed, which can be used to launch all experiments. To see all available configurations, run

```bash
scoresbi -h
```
This command can be used to train and evaluate Simformer variants or baselines, for example by using

```bash
scoresbi method=score_transformer task=two_moons
```
It will train and evaluate a Simformer with default arguments on the two moons dataset.

To run specific experiments, you can use the `experiment` group. This will run a **sweep** over multiple configurations, requiring a specific `launcher` to be specified. By default, the `slurm submitit` launcher is used, which requires SLURM to be installed on your system. You likely need to adjust some configurations in `src/scoresbibm/config/launcher/slurm.yaml` and `src/scoresbibm/config/partition` to your system (to whatever compute resources and partitions exist on your system).

You can also use the `local` launcher to run experiments locally (not recommended for larger experiments).

To run the SBIBM benchmark for all Simformer variants (without baselines) run

```bash
scoresbi +experiment=bm
```

### Creating the plots

In the `figures` folder, we provide notebooks to create the plots from the paper. These notebooks require the experiments (or subsets) to be run first. The experiments needed to create the figure is specified in the README of the respective figure folder.

## Examples

We provide a set of examples to demonstrate the method and its capabilities in the `examples` directory.

This currently includes a
* [Example 1](1_minimal_code_example.ipynb): Minimal code example.<a target="_blank" href="https://colab.research.google.com/github/mackelab/simformer/blob/main/example/1_minimal_code_example.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

* [Example 2](2_two_moons_example.ipynb): Two moons interactive plot + guidance <a target="_blank" href="https://colab.research.google.com/github/mackelab/simformer/blob/main/example/2_two_moons_example.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

* [Example 3](3_slcp_example.ipynb): SLCP interactive plot.<a target="_blank" href="https://colab.research.google.com/github/mackelab/simformer/blob/main/example/3_slcp_example.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


# Citation

If you use this code, please cite the following [paper](https://arxiv.org/abs/2404.09636):

```
@misc{gloeckler2024allinone,
      title={All-in-one simulation-based inference}, 
      author={Manuel Gloeckler and Michael Deistler and Christian Weilbach and Frank Wood and Jakob H. Macke},
      year={2024},
      eprint={2404.09636},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
