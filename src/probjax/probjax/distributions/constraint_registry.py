import jax
from jax import lax

import jax.numpy as jnp

from .constraints import (
    Constraint,
    real,
    integer,
    boolean,
    interval,
    unit_square,
    unit_interval,
    positive,
    strict_positive,
    negative,
    strict_negative,
    positive_integer,
    negative_integer,
    strict_positive_integer,
    strict_negative_integer,
    unit_integer_interval,
    simplex,
    matrix,
    square_matrix,
    positive_definite_matrix,
)

__all__ = [
    "biject_to",
    "transform_to",
]


class ConstraintRegistry:
    """
    Registry to link constraints to transforms.
    """

    def __init__(self):
        self._registry = {}
        super().__init__()

    def register(self, constraint, factory=None):
        """
        Registers a :class:`~torch.distributions.constraints.Constraint`
        subclass in this registry. Usage::

            @my_registry.register(MyConstraintClass)
            def construct_transform(constraint):
                assert isinstance(constraint, MyConstraint)
                return MyTransform(constraint.arg_constraints)

        Args:
            constraint (subclass of :class:`~torch.distributions.constraints.Constraint`):
                A subclass of :class:`~torch.distributions.constraints.Constraint`, or
                a singleton object of the desired class.
            factory (Callable): A callable that inputs a constraint object and returns
                a  :class:`~torch.distributions.transforms.Transform` object.
        """
        # Support use as decorator.
        if factory is None:
            return lambda factory: self.register(constraint, factory)

        # Support calling on singleton instances.
        if isinstance(constraint, Constraint):
            constraint = type(constraint)

        if not isinstance(constraint, type) or not issubclass(constraint, Constraint):
            raise TypeError(
                "Expected constraint to be either a Constraint subclass or instance, "
                "but got {}".format(constraint)
            )

        def factory_wrapper(*args):
            out = jax.tree_util.tree_map(factory, args)
            if len(out) == 1:
                return out[0]
            else:
                return out

        self._registry[constraint] = factory_wrapper

        return factory_wrapper

    def __call__(self, constraint):
        """
        Looks up a transform to constrained space, given a constraint object.
        Usage::

            constraint = Normal.arg_constraints['scale']
            scale = transform_to(constraint)(torch.zeros(1))  # constrained
            u = transform_to(constraint).inv(scale)           # unconstrained

        Args:
            constraint (:class:`~torch.distributions.constraints.Constraint`):
                A constraint object.

        Returns:
            A :class:`~torch.distributions.transforms.Transform` object.

        Raises:
            `NotImplementedError` if no transform has been registered.
        """
        # Look up by Constraint subclass.
        try:
            factory = self._registry[type(constraint)]
        except KeyError:
            raise NotImplementedError(
                f"Cannot transform {type(constraint).__name__} constraints"
            ) from None
        return factory


biject_to = ConstraintRegistry()
transform_to = ConstraintRegistry()


# Register constraints.
def identity(x):
    return x


def generate_matrix(x):
    m = jnp.broadcast_to(jnp.eye(x.shape[-1]), x.shape + (x.shape[-1],))
    return m


def generate_pdm(x):
    m = jnp.broadcast_to(jnp.eye(x.shape[-1]), x.shape + (x.shape[-1],))
    return m


biject_to.register(real)(identity)
transform_to.register(real)(identity)


transform_to.register(integer)(lax.round)
transform_to.register(positive_integer)(lambda x: lax.abs(lax.round(x)))
transform_to.register(negative_integer)(lambda x: -lax.abs(lax.round(x)))
transform_to.register(strict_positive_integer)(
    lambda x: jnp.maximum(lax.abs(lax.round(x)), 1)
)
transform_to.register(strict_negative_integer)(
    lambda x: -jnp.maximum(lax.abs(lax.round(x)), 1)
)

transform_to.register(positive)(lax.abs)
biject_to.register(positive)(lax.exp)

transform_to.register(strict_positive)(
    lambda x: jnp.maximum(lax.abs(x), jnp.finfo(x.dtype).eps)
)
biject_to.register(strict_positive)(lambda x: lax.exp(x) + jnp.finfo(x.dtype).eps)

transform_to.register(negative)(lambda x: -lax.abs(x))
biject_to.register(negative)(lambda x: -lax.exp(x))

transform_to.register(strict_negative)(
    lambda x: -jnp.maximum(lax.abs(x), jnp.finfo(x.dtype).eps)
)
biject_to.register(strict_negative)(lambda x: -lax.exp(x) - jnp.finfo(x.dtype).eps)

transform_to.register(unit_interval)(jax.nn.sigmoid)
biject_to.register(unit_interval)(jax.nn.sigmoid)
transform_to.register(unit_square)(lax.tanh)
biject_to.register(unit_square)(lax.tanh)

transform_to.register(simplex)(jax.nn.softmax)
transform_to.register(matrix)(generate_matrix)
transform_to.register(square_matrix)(generate_matrix)
transform_to.register(positive_definite_matrix)(generate_pdm)
