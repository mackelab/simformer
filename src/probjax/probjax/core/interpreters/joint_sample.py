import jax
from jax.core import JaxprEqn, ClosedJaxpr

from probjax.core.jaxpr_propagation.utils import ForwardProcessingRule
from probjax.core.custom_primitives.random_variable import rv_p
from probjax.core.jaxpr_propagation.interpret import interpret

from typing import Any, Iterable, Sequence, Optional, Tuple


class JointSampleProcessingRule(ForwardProcessingRule):
    joint_samples: dict = {}

    def __init__(self, rvs: Optional[Iterable] = None) -> None:
        """Subset of random variables to be sampled jointly. By default all are sampled!

        Args:
            rvs (Optional[Iterable], optional): Subset of random variable names. Defaults to None.
        """
        self.rvs = rvs
        self.joint_samples = {}

    def __call__(
        self, eqn: JaxprEqn, known_inputs: Sequence[Any | None], _: Sequence[Any | None]
    ) -> Tuple[Sequence[Any | None], Sequence[Any | None]]:
        outvars, outvals = super().__call__(eqn, known_inputs, _)
        if eqn.primitive is rv_p:
            name = eqn.params["name"]
            intervened = eqn.params.get("intervened", False)
            if not intervened and (self.rvs is None or name in self.rvs):
                self.joint_samples[name] = outvals[0]
        return outvars, outvals
