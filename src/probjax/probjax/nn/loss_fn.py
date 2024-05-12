import jax
import jax.numpy as jnp
from jax.random import PRNGKey

from functools import partial

from typing import Optional, Sequence, Union, Callable
from jaxtyping import PyTree, Array


# Score matching objectives


# Flow matching objectives


@partial(jax.jit, static_argnames=("model_fn", "mean_fn", "std_fn"))
def conditional_flow_and_score_matching_loss(
    params: PyTree,
    key: PRNGKey,
    times: Array,
    xs_source: Array,
    xs_target: Array,
    model_fn: Callable,
    mean_fn: Callable,
    std_fn: Callable,
    *args,
    estimate_score: bool = False,
):
    """This function computes the conditional flow matching loss and score matching loss. By setting estimate_score to False, only the conditional flow matching loss is computed. By setting estimate_score to True, both the conditional flow matching loss and score matching loss are computed.


    Args:
        params (PyTree): Parameters of the model_fn given as a PyTree.
        key (PRNGKey): Random key.
        times (Array): Time points, should be broadcastable to shape (batch_size, 1).
        xs_source (Array): Marginal distribution at time t=0, refered to as source distribution.
        xs_target (Array): Marginal distribution at time t, refered to as target distribution.
        model_fn (Callable): Model_fn that takes parameters, times, and samples as input and returns the vector field and optionally the marginal score. Should be a function of the form model_fn(params, times, xs_t) -> v_t(, s_t).
        mean_fn (Callable): The mean function of the Gaussian probability path, should satisfy the following:
                                - mean_fn(xs_source, xs_target, 0) -> xs_source
                                - mean_fn(xs_source, xs_target, 1) -> xs_target
                                - Lipschitz continuous in time
        std_fn (Callable): The standard deviation function of the Gaussian probability path, should satisfy the following:
                                - std_fn(xs_source, xs_target, 0) -> 0
                                - std_fn(xs_source, xs_target, 1) -> 0
                                - std_fn(xs_source, xs_target, t) > 0 for all t in [0, 1]
                                - Two times continuously differentiable in time.
        estimate_score (bool, optional): If set to true, both flow and score matching objectives are computed. Defaults to False.

    Returns:
        (loss_flow, Optional[loss_score]): Respective loss functions
    """
    # Sample x_t
    eps = jax.random.normal(key, shape=xs_source.shape)
    xs_t = (
        mean_fn(xs_source, xs_target, times) + std_fn(xs_source, xs_target, times) * eps
    )

    # Compute u_t -> For flow matching
    # This is valid for Gaussian probability paths, which is currented here.
    t = jnp.broadcast_to(
        times, xs_target.shape
    )  # Pad to x shape for jax.grad -> x.shape
    std_fn_grad = jax.grad(lambda x_s, x_t, t: std_fn(x_s, x_t, t).sum(), argnums=2)
    mean_fn_grad = jax.grad(lambda x_s, x_t, t: mean_fn(x_s, x_t, t).sum(), argnums=2)
    u_t = std_fn_grad(xs_source, xs_target, t) * eps + mean_fn_grad(
        xs_source, xs_target, t
    )

    # Compute loss
    if not estimate_score:
        # Compute vector field -> Flow matching loss
        v_t = model_fn(params, times, xs_t, *args)

        # Compute loss
        loss = jnp.mean(jnp.sum((v_t - u_t) ** 2, axis=-1))

        return loss
    else:
        # Compute vector field and marginal score -> Flow matching loss + Score matching loss
        v_t, s_t = model_fn(params, times, xs_t, *args)

        # Compute loss
        loss = jnp.mean(jnp.sum((v_t - u_t) ** 2, axis=-1))
        loss_score = jnp.mean(
            jnp.sum((s_t + 1 / std_fn(xs_source, xs_target, times) * eps) ** 2, axis=-1)
        )

        return loss, loss_score


def denoising_score_matching_loss(
    params: PyTree,
    key: PRNGKey,
    times: Array,
    xs_target: Array,
    loss_mask: Optional[Array],
    *args,
    model_fn: Callable,
    mean_fn: Callable,
    std_fn: Callable,
    weight_fn: Callable,
    axis: int = -2,
    rebalance_loss: bool = False,
    **kwargs,
) -> Array:
    """This function computes the denoising score matching loss. Which can be used to train diffusion models.

    Args:
        params (PyTree): Parameters of the model_fn given as a PyTree.
        key (PRNGKey): Random generator key.
        times (Array): Time points, should be broadcastable to shape (batch_size, 1).
        xs_target (Array): Target distribution.
        loss_mask (Optional[Array]): Mask for the target distribution. If None, no mask is applied, should be broadcastable to shape (batch_size, 1).
        model_fn (Callable): Score model that takes parameters, times, and samples as input and returns the score. Should be a function of the form model_fn(params, times, xs_t, *args) -> s_t.
        mean_fn (Callable): Mean function of the SDE.
        std_fn (Callable): Std function of the SDE.
        weight_fn (Callable): Weight function for the loss.
        axis (int, optional): Axis to sum over. Defaults to -2.
        

    Returns:
        Array: Loss
    """
    eps = jax.random.normal(key, shape=xs_target.shape)
    mean_t = mean_fn(times, xs_target)
    std_t = std_fn(times, xs_target)
    xs_t = mean_t + std_t * eps
    
    if loss_mask is not None:
        loss_mask = loss_mask.reshape(xs_target.shape)
        xs_t = jnp.where(loss_mask, xs_target, xs_t)
    
    score_pred = model_fn(params, times, xs_t, *args, **kwargs)
    score_target = -eps / std_t

    loss = (score_pred - score_target) ** 2
    if loss_mask is not None:
        loss = jnp.where(loss_mask, 0.0,loss)
    loss = weight_fn(times) * jnp.sum(loss, axis=axis, keepdims=True)
    if rebalance_loss:
        num_elements = jnp.sum(~loss_mask, axis=axis, keepdims=True)
        loss = jnp.where(num_elements > 0, loss / num_elements, 0.0)
    loss = jnp.mean(loss)

    return loss


def score_matching_loss(
    params: PyTree,
    times: Array,
    xs_target: Array,
    mask: Optional[Array],
    *args,
    model_fn: Callable,
):
    """Score matching loss. Minimizing the Fisher divergence between the model and the target distribution, using partial integration trick.

    NOTE: This becomes inefficient when the dimension of the target distribution is high, as the Jacobian of the model is computed.

    Args:
        params (PyTree): Parameters of the model_fn given as a PyTree.
        times (Array): Time points, should be broadcastable to shape (batch_size, 1).
        xs_target (Array): Target distribution.
        mask (Optional[Array]): _description_
        model_fn (Callable): _description_

    Returns:
        _type_: _description_
    """
    jac_model_fn = jax.jacfwd(model_fn, argnums=2)
    score = model_fn(params, times, xs_target, *args)
    jac_score = jac_model_fn(params, times, xs_target, *args)
    loss = jnp.trace(jac_score) + jnp.sum(score**2, axis=-1)
    loss = jnp.mean(jnp.where(mask, loss, 0.0))
    return loss


def sliced_score_matching(
    params: PyTree,
    key: PRNGKey,
    times: Array,
    xs_target: Array,
    mask: Optional[Array],
    *args,
    model_fn: Callable,
    num_slices: int = 1,
):
    def _f(x, v):
        val, grad = jax.value_and_grad(
            lambda x, v: jnp.sum(model_fn(params, times, xs_target, *args) * v)
        )(x, v)
        grad = jnp.sum(grad * v)
        return val, grad

    _f = jax.vmap(jax.vmap(_f, in_axes=(None, 0)), in_axes=(0, None))

    # Slice directions
    v = jax.random.normal(key, shape=(num_slices, xs_target.shape[-1]))
    sliced_score, jac_trace = _f(xs_target, v)
    loss = jnp.sum(sliced_score**2, -1) + jnp.mean(jac_trace, -1)
    loss = jnp.mean(jnp.where(mask, loss, 0.0))
    return loss
