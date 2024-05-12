import jax
import jax.numpy as jnp

from jax.tree_util import tree_flatten, tree_unflatten

from functools import total_ordering

def _precision_to_scale_tril(P):
    Lf = jax.lax.cholesky(jnp.flip(P, (-2, -1)))
    L_inv = jnp.transpose(jnp.flip(Lf, (-2, -1)), (-2, -1))
    Id = jnp.eye(P.shape[-1], dtype=P.dtype, device=P.device)
    L = jax.lax.triangular_solve(L_inv, Id, left_side=False, lower=False)
    return L


@total_ordering
class Match:
    # Subclass ordering...
    __slots__ = ["types"]

    def __init__(self, *types):
        self.types = types

    def __eq__(self, other):
        return self.types == other.types

    def __le__(self, other):
        for x, y in zip(self.types, other.types):
            if not issubclass(x, y):
                return False
            if x is not y:
                break
        return True
