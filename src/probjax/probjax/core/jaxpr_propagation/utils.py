from abc import ABC, abstractmethod

import jax
from jax.core import Jaxpr, JaxprEqn, Literal, Var, Atom, ClosedJaxpr
from jax.tree_util import tree_flatten, tree_unflatten
from jax._src.api_util import (
    flatten_fun_nokwargs,
    argnums_partial_except,
    flatten_fun,
    flatten_fun_nokwargs,
    shaped_abstractify,
)

from jax._src.util import safe_map as map
from typing import Callable, Sequence, Tuple, Union, Optional, Any
from jaxtyping import Array

import math

# High level API


class Environment(dict):
    """A compute environment that stores intermediate computations."""

    def __getitem__(self, var: Atom | None) -> Optional[Array]:
        if isinstance(var, Literal):
            return var.val
        elif var in self:
            return super().__getitem__(var)
        else:
            return None

    def __setitem__(self, var: Atom | None, val: Array | None) -> None:
        if not isinstance(var, Literal):
            super().__setitem__(var, val)

    def read(self, var: Atom | None) -> Array | None:
        return self[var]

    def write(self, var: Atom | None, val: Array | None) -> None:
        self[var] = val

    def known(self, var: Atom | None) -> bool:
        return isinstance(var, Literal) or var in self


class ProcessingRule(ABC):
    """A processing rule for equations."""

    @abstractmethod
    def __call__(
        self,
        eqn: JaxprEqn,
        known_inputs: Sequence[Any | None] | None,
        known_outputs: Sequence[Any | None] | None,
    ) -> Tuple[Sequence[Any | None], Sequence[Any | None]]:
        pass


class ForwardProcessingRule(ProcessingRule):
    def __call__(
        self,
        eqn: JaxprEqn,
        known_inputs: Sequence[Array | None],
        _: Sequence[Array | None],
    ) -> Tuple[Sequence[Atom | None], Sequence[Array | None]]:
        # assert (
        #     (known_inputs != None) and (None not in known_inputs)
        # ), "All inputs must be known for the forward pass."
        primitive = eqn.primitive
        subfuns, bind_params = primitive.get_bind_params(eqn.params)
        # `bind` is how a primitive is called
        outvals = primitive.bind(*subfuns, *known_inputs, **bind_params)
        # Primitives may return multiple outputs or not
        if not eqn.primitive.multiple_results:
            outvals = [outvals]

        return eqn.outvars, outvals  # type: ignore


# Some helper functions


def construct_jaxpr_graph(jaxpr: Jaxpr):
    neighbors = {}

    for i, eqn in enumerate(jaxpr.eqns):
        vars = eqn.invars + eqn.outvars
        for var in vars:
            if not isinstance(var, Literal):
                if var not in neighbors:
                    neighbors[var] = [i]
                else:
                    neighbors[var].append(i)

    return neighbors


def remove_closed_jaxpr_vars_with_suffix(closed_jaxpr, suffix="_"):
    jaxpr = closed_jaxpr.jaxpr
    new_jaxpr = remove_jaxpr_vars_with_suffix(jaxpr, suffix=suffix)
    return ClosedJaxpr(new_jaxpr, closed_jaxpr.literals)


def remove_jaxpr_vars_with_suffix(jaxpr, suffix="_"):
    return jaxpr.replace(invars=[v for v in jaxpr.invars if v.suffix != suffix])


def jaxpr_returning_const(*consts, invars=[]):
    consts, const_tree = tree_flatten(consts)
    const_avals = tuple(map(shaped_abstractify, consts))
    const_vars = [Var(0, "_obs", c_aval) for c_aval in const_avals]
    new_jaxpr = Jaxpr(const_vars, invars, const_vars, [])
    new_closed_jaxpr = ClosedJaxpr(new_jaxpr, consts)
    return new_closed_jaxpr, const_tree
