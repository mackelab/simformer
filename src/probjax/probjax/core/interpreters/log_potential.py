import math

import jax
from jax.core import JaxprEqn, Jaxpr, eval_jaxpr
from jaxtyping import Array

from probjax.core.jaxpr_propagation.utils import ForwardProcessingRule
from probjax.core.custom_primitives.random_variable import rv_p

from typing import Callable, Sequence, Optional, Dict


def potential_cost_fn(
    eqn: JaxprEqn, in_known: Sequence[bool], out_known: Sequence[bool]
):
    if eqn.primitive is rv_p and all(in_known):
        # Process random variables first (if there are only random variables we can skip the rest)
        return 2
    elif all(in_known):
        # If one random variable is parameterized by a previous, then we must process the "inbetween" computations
        return 1
    elif eqn.primitive is rv_p and eqn.params.get("intervened", False):
        # If the random variable is intervened, we must process it
        return 0
    else:
        # This should never happen
        return math.inf


def extract_random_vars_values(jaxpr: Jaxpr, joint_samples: Dict[str, Array]):
    vars = []
    values = []

    for eqn in jaxpr.eqns:
        if eqn.primitive is rv_p:
            name = eqn.params["name"]
            intervened = eqn.params["intervened"]
            if not intervened:
                vars.append(eqn.outvars[0])
                values.append(joint_samples[name])
            else:
                vars.extend(eqn.invars)
                values.extend([jax.numpy.zeros(shape=v.aval.shape, dtype=v.aval.dtype) for v in eqn.invars])
            
            

    return vars, values


class LogPotentialProcessingRule(ForwardProcessingRule):
    # Here we accumlate the joint_log_prob of all the random variables!
    log_prob: float = 0.0
    joint_samples: Dict[str, Array]

    def __init__(self, joint_samples: Dict[str, Array]):
        self.joint_samples = joint_samples
        #print(self.joint_samples)
    def __call__(
        self,
        eqn: JaxprEqn,
        in_known: Sequence[Array | None],
        out_known: Sequence[Array | None],
    ):
        if eqn.primitive is rv_p:
            # We do not have to sample -> Already given
            name = eqn.params["name"]
            intervened = eqn.params.get("intervened", False)
            if not intervened:
                outvars = eqn.outvars
                outvals = [self.joint_samples[name]]
            else:
                outvars, outvals = super().__call__(eqn, in_known, out_known)
            # But we still have to compute the log_prob
            in_known = list(in_known)
            #print(name, outvals)
            in_known[-1] = outvals[0] # From where do I know this?
            log_prob_fn = eqn.params["log_prob_fn_jaxpr"]
            self.log_prob += eval_jaxpr(
                log_prob_fn.jaxpr, log_prob_fn.consts, *in_known
            )[0]
        else:
            outvars, outvals = super().__call__(eqn, in_known, out_known)

        return outvars, outvals
