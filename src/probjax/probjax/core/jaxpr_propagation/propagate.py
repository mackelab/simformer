import jax
from jax.core import Jaxpr, JaxprEqn, Literal, Var, Atom
from jax.experimental.pjit import pjit_p
from jax.custom_derivatives import custom_jvp_call_p
from jax._src.util import safe_map as map
from typing import Callable, Sequence, Tuple, Sequence, Optional, Any
from jaxtyping import Array

import math

from probjax.utils.containers import PriorityQueue
from probjax.core.jaxpr_propagation.utils import (
    Environment,
    construct_jaxpr_graph,
    ForwardProcessingRule,
    ProcessingRule,
)


# Jaxpr are usually processed in a topological order. But we need to process it in a custom order.
# Hence this gives use a general propagation algorithm!


def naive_cost_fn(
    eqn: JaxprEqn, is_known_invars: Sequence[bool], is_known_outvars: Sequence[bool]
) -> float:
    """A naive cost function that returns 0 if all inputs are known and inf otherwise.
    This cost function is used for the forward pass. We can only compute the equation if all inputs are known.

    Args:
        eqn (JaxprEqn): JaxprEqn object.
        is_known_invars (Sequence[bool]): Sequence of booleans indicating whether the input variables are known.
        is_known_outvars (Sequence[bool]): Sequence of booleans indicating whether the output variables are known.

    Returns:
        float: Cost of the equation.
    """
    if all(is_known_invars):
        return 0.0
    else:
        return math.inf


class EqnEnvironment:
    """This class implements a priority queue for equations. The priority is determined by the cost function."""

    env: Environment
    eqn_queue: PriorityQueue

    def __init__(
        self,
        G: dict,
        env: Environment,
        eqns: Sequence[JaxprEqn],
        cost_fn: Callable = naive_cost_fn,
    ):
        self.G = G
        self.env = env
        self.eqns = eqns
        self.eqn_queue = PriorityQueue()
        self.processed_eqns = set()
        self.cost_fn = cost_fn

        self._init_eqn_queue()

    def _init_eqn_queue(self):
        for v in self.env:
            eqns_index = self.G[v]
            map(self.write, eqns_index)

        for i, eqn in enumerate(self.eqns):
            if all(map(self.env.known, eqn.invars)):
                self.write(i)

    def write(self, index: int):
        if index in self.processed_eqns:
            # Do not write processed equations
            return

        if index in self.eqn_queue:
            # Remove equation from the queue if it is already present
            eqn = self.eqns[index]
            known_invars = map(self.env.known, eqn.invars)
            known_outvars = map(self.env.known, eqn.outvars)
            cost = self.cost_fn(eqn, known_invars, known_outvars)
            self.eqn_queue.update_cost(index, cost)
            return
        # Write equation to the queue
        eqn = self.eqns[index]
        known_invars = map(self.env.known, eqn.invars)
        known_outvars = map(self.env.known, eqn.outvars)
        cost = self.cost_fn(eqn, known_invars, known_outvars)
        self.eqn_queue.insert(index, cost)

    def pop(self) -> JaxprEqn:
        index = self.eqn_queue.pop()
        # print(index)
        eqn = self.eqns[index]
        self.processed_eqns.add(index)
        return eqn

    def is_empty(self) -> bool:
        return self.eqn_queue.is_empty()


# TODO: We have to add call this recursively for nested jaxprs.


def propagate(
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
    cost_fn: Callable = naive_cost_fn,
    process_all_eqns: bool = False,
) -> Sequence[Optional[Array]]:
    """This implements a general propagation algorithm for JaxPr.

    Args:
        jaxpr (Jaxpr): Jaxpr object.
        consts (Sequence[Array]): Constants.
        invars (Sequence[Var]): Input variables.
        inputs (Sequence[Array]): Input data.
        outvars (Sequence[Var]): Output variables.
        process_eqn (Callable[ [JaxprEqn, Sequence[Optional[Array]], Sequence[Optional[Array]]], Tuple[Sequence[Var], Sequence[Array]], ]): Function that processes the equation.
        cost_fn (Callable, optional): Cost function to prioritize the evaluation order!. Defaults to naive_cost_fn.

    Returns:
        Sequence[Optional[Array]]: _description_
    """

    # Set up environment
    env = Environment()

    G = construct_jaxpr_graph(jaxpr)

    map(env.write, jaxpr.constvars, consts)
    map(env.write, invars, inputs)

    # Set up equations
    eqn_env = EqnEnvironment(G, env, jaxpr.eqns, cost_fn)

    while not eqn_env.is_empty():
        # print(list(env.keys()))
        # print(eqn_env.eqn_queue)

        eqn = eqn_env.pop()  # Equation to process

        # Read known invars and outvars
        known_invars = map(env.read, eqn.invars)
        known_outvars = map(env.read, eqn.outvars)

        if not all([v is not None for v in known_invars]) and eqn.primitive is pjit_p:
            if eqn.primitive is pjit_p:
                closed_sub_jaxpr = eqn.params["jaxpr"]
            else:
                closed_sub_jaxpr = eqn.params["call_jaxpr"]
            sub_jaxpr = closed_sub_jaxpr.jaxpr
            sub_consts = closed_sub_jaxpr.consts

            sub_invars = []
            sub_invar_vals = []
            sub_outvars = []
            for v, val in zip(
                sub_jaxpr.invars + sub_jaxpr.outvars, known_invars + known_outvars
            ):
                if val is None:
                    sub_outvars.append(v)
                else:
                    sub_invars.append(v)
                    sub_invar_vals.append(val)

            output_vals = propagate(
                sub_jaxpr,
                sub_consts,
                sub_invars,
                sub_invar_vals,
                sub_outvars,
                process_eqn=process_eqn,
                cost_fn=cost_fn,
                process_all_eqns=process_all_eqns,
            )
            ouput_vars = []
            for v, val in zip(eqn.invars + eqn.outvars, known_invars + known_outvars):
                if val is None:
                    ouput_vars.append(v)

            process_eqn(eqn, known_invars, ouput_vars)

        else:
            # This processes the computation
            ouput_vars, output_vals = process_eqn(eqn, known_invars, known_outvars)

        # Write outvars
        map(env.write, ouput_vars, output_vals)

        # Update reachable equations
        for v in ouput_vars:
            eqns_index = G[v]
            map(eqn_env.write, eqns_index)

        if not process_all_eqns and all(map(env.known, outvars)):
            break

        # print(list(env.keys()))
        # print(eqn_env.eqn_queue)

    # Gather target values
    out = map(env.read, outvars)

    return out
