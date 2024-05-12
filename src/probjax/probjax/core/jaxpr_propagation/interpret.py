import jax

from jax.core import Jaxpr, JaxprEqn, Literal, Var, Atom
from jax.experimental.pjit import pjit_p
from jax._src.util import safe_map as map
from typing import Callable, Sequence, Tuple, Union, Optional, Any
from jaxtyping import Array

import math

from probjax.utils.containers import PriorityQueue
from probjax.core.jaxpr_propagation.utils import (
    Environment,
    construct_jaxpr_graph,
    ProcessingRule,
    ForwardProcessingRule,
)

# Simple base interpreter


def interpret(
    jaxpr: Jaxpr,
    consts: Sequence[Array],
    invars: Sequence[Var],
    inputs: Sequence[Array],
    outvars: Sequence[Var],
    process_eqn: ProcessingRule
    | Callable[
        [JaxprEqn, Sequence[Optional[Array]], Sequence[Optional[Array]]],
        Tuple[Sequence[Var], Any],
    ] = ForwardProcessingRule(),
):
    env = Environment()

    # Initialize the environment with the constants and inputs
    map(env.write, jaxpr.constvars, consts)
    map(env.write, invars, inputs)

    # We iterate in one (of many other) topological ordering of the graph. The topological sort is already done by JAX.
    for eqn in jaxpr.eqns:
        # First we get the known inputs and outputs of the equation
        known_invars = map(env.read, eqn.invars)
        known_outvars = map(env.read, eqn.outvars)

        # Higher level primitives i.e. jit requires recursive processing of the equation
        if eqn.primitive is pjit_p:
            sub_jaxpr = eqn.params["jaxpr"]

            sub_invars = []
            sub_invar_vals = []
            sub_outvars = []
            for v, val in zip(
                sub_jaxpr.jaxpr.invars + sub_jaxpr.jaxpr.outvars, known_invars + known_outvars
            ):
                if val is None:
                    sub_outvars.append(v)
                else:
                    sub_invars.append(v)
                    sub_invar_vals.append(val)

            outvals = interpret(
                sub_jaxpr.jaxpr,
                sub_jaxpr.consts,
                sub_invars,
                sub_invar_vals,
                sub_outvars,
                process_eqn,
            )
            outvar = eqn.outvars
            process_eqn(eqn, known_invars, known_outvars)
        else:
            # We process the equation
            outvar, outvals = process_eqn(eqn, known_invars, known_outvars)  # type: ignore

        # Write outvars
        map(env.write, outvar, outvals)

        # TODO Maybe clean up environments!

    # Gather target values
    out = map(env.read, outvars)

    return out
