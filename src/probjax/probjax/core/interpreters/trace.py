import jax
from jax.core import JaxprEqn, ClosedJaxpr

from probjax.core.jaxpr_propagation.utils import ForwardProcessingRule
from probjax.core.custom_primitives.random_variable import rv_p
from probjax.core.jaxpr_propagation.interpret import interpret

from typing import Any, Iterable, Sequence, Optional, Tuple


class TraceProcessingRule(ForwardProcessingRule):
    traced_samples: dict = {}

    def __init__(self, traced_vars: Optional[Iterable] = None) -> None:
        """Subset of random variables to be sampled jointly. By default all are sampled!

        Args:
            rvs (Optional[Iterable], optional): Subset of random variable names. Defaults to None.
        """
        self.traced_vars = traced_vars

    def __call__(
        self, eqn: JaxprEqn, known_inputs: Sequence[Any | None], _: Sequence[Any | None]
    ) -> Tuple[Sequence[Any | None], Sequence[Any | None]]:
        outvars, outvals = super().__call__(eqn, known_inputs, _)
        for o, v in zip(outvars, outvals):
            if self.traced_vars is None or str(o) in self.traced_vars:
                self.traced_samples[str(o)] = v
        return outvars, outvals

class TraceRandomRule(TraceProcessingRule):
    traced_rvs: dict = {}
    # Trace all random variables
    def __init__(self, traced_vars: Iterable | None = None) -> None:
        super().__init__(traced_vars)
