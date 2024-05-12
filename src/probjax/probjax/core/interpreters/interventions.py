import jax
from jax.core import JaxprEqn, ClosedJaxpr

from probjax.core.jaxpr_propagation.utils import ForwardProcessingRule
from probjax.core.custom_primitives.random_variable import rv_p
from probjax.core.jaxpr_propagation.interpret import interpret

from typing import Any, Iterable, Sequence, Optional, Tuple
from jaxtyping import Array


class IntervenedProcessingRule(ForwardProcessingRule):
    interventions: dict = {}

    def __init__(self, interventions: dict[str, Array]) -> None:
        """Subset of random variables to be sampled jointly. By default all are sampled!

        Args:
            rvs (Optional[Iterable], optional): Subset of random variable names. Defaults to None.
        """
        self.interventions = interventions

    def __call__(
        self, eqn: JaxprEqn, known_inputs: Sequence[Any | None], _: Sequence[Any | None]
    ) -> Tuple[Sequence[Any | None], Sequence[Any | None]]:
        if eqn.primitive is rv_p:
            name = eqn.params["name"]
            if name in self.interventions:
                new_sampling_jaxpr = jax.make_jaxpr(
                    lambda *args: self.interventions[name]
                )(*known_inputs)
                eqn.params["sampling_fn_jaxpr"] = new_sampling_jaxpr
                eqn.params["intervened"] = True

            outvars, outvals = super().__call__(eqn, known_inputs, _)
        else:
            outvars, outvals = super().__call__(eqn, known_inputs, _)
        return outvars, outvals
