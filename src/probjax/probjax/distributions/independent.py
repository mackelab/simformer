import jax.numpy as jnp
from jax.scipy.special import logsumexp

import jax
from jax import random
from jax.lax import scan

from typing import Optional, Sequence, Union

from .distribution import Distribution
from .constraints import real, positive, unit_interval

__all__ = ["Independent"]

from jax.tree_util import register_pytree_node_class

# Transforms a batch of independent distributions into a single mulitvariate product distribution.


@register_pytree_node_class
class Independent(Distribution):
    """
    Creates an independent distribution by treating the provided distribution as
    a batch of independent distributions.

    Args:
        base_dist: Base distribution object.
        reinterpreted_batch_ndims: The number of batch dimensions that should
            be considered as event dimensions.
    """

    def __init__(
        self,
        base_dist: Union[Distribution, Sequence[Distribution]],
        reinterpreted_batch_ndims: int,
    ):
        # Determine batch_shape and event_shape using the helper function
        batch_shape, event_shape, event_ndims, reinterpreted_batch_ndims = determine_shapes(
            base_dist, reinterpreted_batch_ndims
        )
        

        if isinstance(base_dist, Distribution):
            # Single distribution case
            self.base_dist = [base_dist]
        else:
            self.base_dist = base_dist

        self.event_ndims = event_ndims
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims
        
        for p in self.base_dist:
            p._batch_shape = batch_shape
            p._event_shape = event_shape

        super().__init__(batch_shape=batch_shape, event_shape=event_shape)

    @property
    def mean(self):
        return jnp.stack([b.mean for b in self.base_dist], axis=-1).reshape(
            self.batch_shape + self.event_shape
        )

    @property
    def median(self):
        return jnp.stack([b.median for b in self.base_dist], axis=-1).reshape(
            self.batch_shape + self.event_shape
        )

    @property
    def mode(self):
        # The mode does change and is not equal to the mode of the base distribution
        raise NotImplementedError()

    @property
    def variance(self):
        return jnp.stack([b.variance for b in self.base_dist], axis=-1).reshape(
            self.batch_shape + self.event_shape
        )

    def rsample(self, key, sample_shape=()):
        keys = random.split(key, len(self.base_dist))
        samples = jnp.stack(
            [p.rsample(k, sample_shape) for k, p in zip(keys, self.base_dist)],
            axis=-1,
        )
        return jnp.reshape(samples, sample_shape + self.batch_shape + self.event_shape)

    def sample(self, key, sample_shape=()):
        keys = random.split(key, len(self.base_dist))
        if self.reinterpreted_batch_ndims > 0:
            samples = jnp.hstack(
                [p.sample(k, sample_shape) for k, p in zip(keys, self.base_dist)],
            )
        else:
            samples = jnp.stack(
                [p.sample(k, sample_shape) for k, p in zip(keys, self.base_dist)],
                axis=-len(self.event_shape) - 1,
            )
        return jnp.reshape(samples, sample_shape + self.batch_shape + self.event_shape)

    def log_prob(self, value):
        if len(self.base_dist) == 1:
            log_prob = self.base_dist[0].log_prob(value)
        else:
            if self.reinterpreted_batch_ndims > 0:
                split_value = jnp.split(value,self.event_ndims, axis=-1)[1:]
                log_prob = jnp.stack(
                    [
                        b.log_prob(v.reshape((-1,) + b.batch_shape + b.event_shape))
                        for b, v in zip(self.base_dist, split_value)
                    ], axis=-1
                )
                log_prob = jnp.reshape(
                    log_prob, value.shape[:-1] + self.batch_shape + (len(self.base_dist),)
                )
            else:
                split_value = jnp.split(
                    value, self.event_ndims, axis=-len(self.event_shape) - 1
                )[1:]
                log_prob = jnp.stack(
                    [b.log_prob(v) for b, v in zip(self.base_dist, split_value)],
                    axis=-len(self.event_shape) - 1,
                )
                log_prob = jnp.reshape(
                    log_prob,
                    value.shape[: -len(self.event_shape) - 1] + self.batch_shape,
                )

        # Sum the log probabilities along the event dimensions
        if self.reinterpreted_batch_ndims > 0:
            return jnp.sum(
                log_prob, axis=tuple(range(-self.reinterpreted_batch_ndims, 0))
            )
        else:
            return log_prob

    def entropy(self):
        entropy = jnp.stack([b.entropy() for b in self.base_dist], axis=-1)

        # Sum the entropies along the event dimensions
        if self.reinterpreted_batch_ndims > 0:
            return jnp.sum(
                entropy, axis=tuple(range(-self.reinterpreted_batch_ndims, 0))
            )
        else:
            return entropy

    def __repr__(self) -> str:
        return f"Independent({self.base_dist}, reinterpreted_batch_ndims={self.reinterpreted_batch_ndims})"

        # Each distribution will be registered as a PyTree

    def tree_flatten(self):
        flat_components, tree_components = jax.tree_util.tree_flatten(self.base_dist)
        return (
            tuple(flat_components),
            [tree_components, self.reinterpreted_batch_ndims],
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        tree_components, reinterpreted_batch_ndims = aux_data
        return cls(
            jax.tree_util.tree_unflatten(tree_components, children),
            reinterpreted_batch_ndims,
        )


def determine_shapes(
    base_dist: Union[Distribution, Sequence[Distribution]],
    reinterpreted_batch_ndims: int,
):
    if isinstance(base_dist, Distribution):
        # Single distribution case
        base_dist = [base_dist]

    # Extract batch shapes and event shapes from the list of base distributions
    batch_shapes = [b.batch_shape for b in base_dist]
    event_shapes = [b.event_shape for b in base_dist]
    
    assert all(reinterpreted_batch_ndims <= len(b) for b in batch_shapes) or all(reinterpreted_batch_ndims <= len(e) for e in event_shapes), "reinterpreted_batch_ndims must be greater than or equal to the batch shape of the base distribution."

    # Ensure that batch shapes are equal and calculate event_shape
    batch_shape, event_shape, event_ndims = calculate_shapes(
        batch_shapes, event_shapes, reinterpreted_batch_ndims
    )

    return tuple(batch_shape), tuple(event_shape), tuple(event_ndims), reinterpreted_batch_ndims


def calculate_shapes(batch_shapes, event_shapes, reinterpreted_batch_ndims):
    event_shape = list(event_shapes[0])
    if reinterpreted_batch_ndims > 0:
        new_event_shape = list(batch_shapes[0][-reinterpreted_batch_ndims:])
        if len(new_event_shape) > 0:
            for b in batch_shapes[1:]:
                if len(b) > 0:
                    new_event_shape[-1] += b[- 1]
                else:
                    new_event_shape[-1] += 1

        if len(event_shape) > 0:
            for e in event_shapes[1:]:
                if len(e) > 0:
                    event_shape[-1] += e[-1]
                else:
                    event_shape[-1] += 1

        batch_shape = tuple(batch_shapes[0][:-reinterpreted_batch_ndims])
        event_shape = tuple(new_event_shape) + tuple(event_shape)
        event_ndims = [0]
        for e in event_shapes:
            if len(e) == 0:
                event_ndims.append(event_ndims[-1] + 1)
            else:
                event_ndims.append(event_ndims[-1] + e[-1])
    else:
        new_batch_shape = list(batch_shapes[0])
        if len(new_batch_shape) > 0:
            for b in batch_shapes[1:]:
                if len(b) > 0:
                    new_batch_shape[-1] += b[-1]
                else:
                    new_batch_shape[-1] += 1
        else:
            new_batch_shape = (len(batch_shapes),)
        batch_shape = tuple(new_batch_shape)
        event_shape = tuple(event_shape)
        event_ndims = [0]
        for b in batch_shapes:
            if len(b) == 0:
                event_ndims.append(event_ndims[-1] + 1)
            else:
                event_ndims.append(event_ndims[-1] + b[-1])

    return batch_shape, event_shape, event_ndims
