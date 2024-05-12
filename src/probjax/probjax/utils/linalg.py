import jax
import jax.numpy as jnp
from jax import lax

from jaxtyping import Array, Float
from typing import Tuple

from jax.scipy.linalg import expm


def is_matrix(A: Array) -> bool:
    """Check if input is a matrix

    Args:
        A (Array): Input array

    Returns:
        bool: True if A is a matrix, or a batch of matrices
    """
    return len(A.shape) >= 2


def is_diagonal_matrix(A: Array, axis1=-2, axis2=-1) -> bool:
    """Check if input is a diagonal matrix

    Args:
        A (Array): Input array

    Returns:
        bool: True if A is a diagonal matrix, or a batch of diagonal matrices
    """
    return is_matrix(A) and jnp.all(
        A == jnp.diag(jnp.diagonal(A, axis1=axis1, axis2=axis2)), axis=(axis1, axis2)
    )


def is_triangular_matrix(A: Array, lower: bool = True) -> bool:
    """Check if input is a triangular matrix

    Args:
        A (Array): Input array
        lower (bool, optional): True if lower triangular. Defaults to True.

    Returns:
        bool: True if A is a triangular matrix, or a batch of triangular matrices
    """
    return is_matrix(A) and jnp.all(
        A == jnp.tril(A) if lower else jnp.triu(A), axis=(-2, -1)
    )


def batch_mv(bmat: Array, bvec: Array) -> Array:
    """
    Performs a batched matrix-vector product, with compatible but different batch shapes.

    This function takes as input `bmat`, containing n x n matrices, and
    `bvec`, containing length n vectors.

    Both `bmat` and `bvec` may have any number of leading dimensions, which correspond
    to a batch shape. They are not necessarily assumed to have the same batch shape,
    just ones which can be broadcasted.
    """
    return jnp.matmul(bmat, bvec[..., jnp.newaxis])[..., 0]


def batch_mahalanobis(bL: Array, bx: Array) -> Array:
    """
    Computes the squared Mahalanobis distance x^T M^-1 x for a factored M = LL^T.

    Accepts batches for both bL and bx. They are not necessarily assumed to have the same batch
    shape, but `bL` one should be able to broadcasted to `bx` one.
    """
    bL = jnp.broadcast_to(bL, bx.shape[:-1] + bL.shape[-2:])
    
    sol = lax.linalg.triangular_solve(bL, bx, lower=True, transpose_a=True)
    return jnp.sum(sol**2, axis=-1)


def transition_matrix(A: Array, t0: Float, t1: Float) -> Array:
    """Transition matrix

    Args:
        A (Array): Drift matrix
        t (float): New time point
        t0 (float): Old time point

    Returns:
        Array: Transition matrix
    """
    if A.shape[-1] == 1:
        return jnp.exp(A * (t1 - t0))
    else:
        return expm(A * (t1 - t0))


def matrix_fraction_decomposition(
    t0: Float, t1: Float, A: Array, B: Array
) -> Tuple[Array, Array]:
    """Matrix fraction decomposition

    Returns the transition matrix and covariance. Is exact if A and B are truely time independent

    Args:
        t0 (float): New time point
        t1 (float): Old time point
        A (Array): Drift matrix
        B (Array): Diffusion matrix

    Returns:
        Tuple[Array]: Transition matrix and covariance
    """
    d = A.shape[-1]
    blockmatrix = jnp.block([[A, jnp.dot(B, B.T)], [jnp.zeros((d, d)), -A.T]])
    M = expm(blockmatrix * (t1 - t0))
    Phi = M[:d, :d]
    Q = jnp.dot(M[:d, d:], Phi.T)
    return Phi, Q
