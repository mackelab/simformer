
## Probjax

Simple and highly experimental backend for some probabilistic computations and neural networks in JAX. 

This implements a few inference backends, neural network architectures, ode/sde solvers and some other utilities. Which
are used in the `scoresbibm` repository.

### Installation

If you have cuda installed, use the following command to install the package:

```bash
pip install -e src/probjax[cuda]
```