import haiku as hk
import jax
import jax.numpy as jnp

from haiku.nets import MLP
from jaxtyping import Array, PyTree

from typing import Callable, Any, List

from probjax.core.custom_primitives.custom_inverse import custom_inverse
from probjax.core.transformation import inverse_and_logabsdet, inverse


def autoregressive_mask_getter(d: int, first_layer: bool = False):
    """Custom getter for autoregressive masks.

    Args:
        d (int): Dimension of the input
        last_layer (bool, optional): Whether this is the last layer. Defaults to False.

    Returns:
        Callable: Getter function
    """

    def getter(next_getter: Callable, value: Array, context: PyTree):
        shape = context.original_shape
        name = context.full_name
        module = context.module
        if isinstance(module, hk.Linear):
            if "/w" in name:
                input_dim = shape[0]
                x1 = jnp.arange(input_dim).reshape(-1, 1) % d + 1
                x2 = jnp.arange(shape[-1]).reshape(1, -1) % d + 1

                if not first_layer:
                    mask = x2 >= x1
                else:
                    mask = x2 > x1

                return next_getter(value * mask)
            else:
                return next_getter(value)
        else:
            raise NotImplementedError("Only Linear layers are supported, currently")

    return getter


class MaskedMLP(MLP):
    def __init__(
        self, custom_mask_getter: Callable, output_sizes: List[int], *args, **kwargs
    ):
        """Haiku MLP, but weights are masked by a custom getter.

        Args:
            custom_mask_getter (Callable): Custom getter
            output_sizes (List[int]): Output sizes of the MLP
            args: args for MLP
            kwargs: kwargs for MLP
        """
        super().__init__(output_sizes, *args, **kwargs)
        self.custom_mask_getter = custom_mask_getter
        self.context_layer = hk.Linear(output_sizes[0])

    def __call__(self, inputs: Array, context: Array | None = None, rng=None) -> Array:
        num_layers = len(self.layers)
        in_dim = inputs.shape[-1]
        out = inputs
        for i, layer in enumerate(self.layers):
            # Masking the weights to be autoregressive (at selected dimensions)!
            with hk.custom_getter(
                self.custom_mask_getter(
                    in_dim,
                    first_layer=i == 0,
                )
            ):
                out = layer(out)

                if i < (num_layers - 1) or self.activate_final:
                    out = self.activation(out)
            # Not masked!
            if i == 0 and context is not None:
                out += self.context_layer(context)
        return out


def autoregressive_transform(
    bijector: Callable, input_dim: int, output_sizes: List[int], *args, **kwargs
):
    # Autoregressive transformation accelerated by MADE
    @hk.without_apply_rng
    @hk.transform
    def forward(x, context: Array | None = None):
        conditionor = MaskedMLP(
            autoregressive_mask_getter, output_sizes, *args, **kwargs
        )
        params = conditionor(x, context)
        y = bijector(params, x)
        return y

    # The inverse now however must be done sequentially, and is provided through a custom_inverse primitive
    @hk.without_apply_rng
    @hk.transform
    def inv(y, context: Array | None = None):
        conditionor = MaskedMLP(
            autoregressive_mask_getter, output_sizes, *args, **kwargs
        )
        x = jnp.ones(y.shape[:-1] + (input_dim,))
        log_det = 0.0
        for _ in range(input_dim):
            params = conditionor(x, context)  # type: ignore
            bijective_inv = inverse_and_logabsdet(lambda x: bijector(params, x))
            x, log_det = bijective_inv(y)
        return x, log_det

    init_fn, apply_fn = forward.init, forward.apply
    _, apply_inv = inv.init, inv.apply

    # Defining the custom inverse primitive
    fun = custom_inverse(apply_fn)
    fun.definv_and_logdet(apply_inv)

    return init_fn, fun


class AutoregressiveMLP:
    def __init__(
        self,
        bijector: Callable,
        num_bijector_params: int,
        hidden_dims: List[int] = [50, 50],
        **kwargs
    ):
        self.output_sizes = hidden_dims + [num_bijector_params]
        self.conditionor = MaskedMLP(
            autoregressive_mask_getter, self.output_sizes, **kwargs
        )
        self.bijector = bijector
        self.num_bijector_params = num_bijector_params

    def __call__(self, inputs: Array, context: Array | None = None, rng=None) -> Array:
        init_rng = hk.next_rng_keys(1)[0] if hk.running_init() else None
        input_dim = inputs.shape[-1]
        # Autoregressive transformation accelerated by MADE
        init_fn, apply_fn = autoregressive_transform(
            self.bijector, input_dim, self.output_sizes
        )
        # We have to lift it as it has its own init function
        # Doing this within here will cause an error, as jax primitives will then be called inside the hk.transform which transforms this module later on.
        init = hk.lift(init_fn)

        # After initialization we can savely call the function
        def f(x: Array, context: Array | None = None):
            params = init(init_rng, x, context)
            return apply_fn(params, x, context)

        return f(inputs, context)
