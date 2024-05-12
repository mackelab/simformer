import haiku as hk
import jax
import jax.numpy as jnp

from typing import Callable, Any, List, Optional
from functools import partial
from jaxtyping import Array, PyTree

import math

from probjax.core.custom_primitives.custom_inverse import custom_inverse


class Flip(hk.Module):
    def __init__(self, axis: int = -1, name: str = "flip"):
        """Flip the array along an axis.

        Args:
            axis (int, optional): Axis to flip. Defaults to -1.
            name (str, optional): Name of the module. Defaults to "flip".
        """
        super().__init__(name=name)
        self.axis = axis

    def __call__(self, x: Array, *args) -> Array:
        return jnp.flip(x, axis=self.axis)


class Permute(hk.Module):
    def __init__(self, permutation: Array, axis: int = -1, name: str = "permute"):
        """Permutes the array along an axis.

        Args:
            permutation (Array): An array of indices to permute.
            axis (int, optional): Axis to permute. Defaults to -1.
            name (str, optional): _description_. Defaults to "permute".
        """
        super().__init__(name=name)
        self.permutation = permutation
        self.axis = axis

    def __call__(self, x: Array, *args) -> Array:
        return jnp.take(x, self.permutation, axis=self.axis)


@partial(custom_inverse, inv_argnum=1)
def rotate(R, x):
    return jnp.matmul(R, x.T).T


rotate.definv_and_logdet(lambda R, x: (jnp.matmul(R.T, x.T).T, 0.0))


class Rotate(hk.Module):
    def __init__(self, key: Array, output_dim: int, name: str = "rotate"):
        """Rotate the array.

        Args:
            rotation_matrix (Array): Rotation matrix.
            name (str, optional): Name of the module. Defaults to "rotate".
        """
        super().__init__(name=name)
        self.rotation_matrix = jax.random.orthogonal(key, output_dim)

    def __call__(self, x: Array, *args) -> Array:
        return rotate(self.rotation_matrix, x)


class SinusoidalEmbedding(hk.Module):
    def __init__(self, output_dim: int = 128, name: str = "sinusoidal_embedding"):
        """Sinusoidal embedding module. Mostly used to embed time.

        Args:
            output_dim (int, optional): Output dimesion. Defaults to 128.
            name (str, optional): Name of the module. Defaults to "sinusoidal_embedding".
        """
        super().__init__(name=name)
        self.output_dim = output_dim

    def __call__(self, inputs):
        half_dim = self.output_dim // 2 + 1
        emb = jnp.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = inputs[..., None] * emb[None, ...]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], -1)
        out = jnp.squeeze(emb, axis=-2)
        return out[..., : self.output_dim]


class GaussianFourierEmbedding(hk.Module):
    def __init__(
        self,
        output_dim: int = 128,
        learnable=False,
        name: str = "gaussian_fourier_embedding",
    ):
        """Gaussian Fourier embedding module. Mostly used to embed time.

        Args:
            output_dim (int, optional): Output dimesion. Defaults to 128.
            name (str, optional): Name of the module. Defaults to "gaussian_fourier_embedding".
        """
        super().__init__(name=name)
        self.output_dim = output_dim
        self.learnable = learnable

    def __call__(self, inputs):
        half_dim = self.output_dim // 2 + 1
        B = hk.get_parameter(
            "B", [half_dim, inputs.shape[-1]], init=hk.initializers.RandomNormal()
        )
        if not self.learnable:
            B = jax.lax.stop_gradient(B)
        term1 = jnp.cos(2 * jnp.pi * jnp.dot(inputs, B.T))
        term2 = jnp.sin(2 * jnp.pi * jnp.dot(inputs, B.T))
        out = jnp.concatenate([term1, term2], axis=-1)
        return out[..., : self.output_dim]


class OneHot(hk.Module):
    """One hot encoding module."""

    num_tokens: int  # Size of the vocabulary.
    name: str | None = None  # Optional identifier for the module.

    def __init__(self, num_tokens: int, name: str | None = "one_hot_embed"):
        """_summary_

        Args:
            num_tokens (int): Number of distinct tokens.
            name (str | None, optional): Name of the module. Defaults to "one_hot_embed".
        """
        super().__init__(name=name)
        self.num_tokens = num_tokens

    def __call__(self, x: Array, rng=None) -> Array:
        """One hot encodes the input.

        Args:
            x (jax.Array): Input array of shape [B, T]
        """
        return jax.nn.one_hot(x, self.num_tokens)


class PosEmbed(hk.Module):
    def __init__(self, token_dim: int, max_seq_len: int = 500):
        """Positional embedding module.

        Args:
            token_dim (int): Dimension of the token embedding.
            max_seq_len (int, optional): Maximal length of the sequence. Defaults to 500.
        """
        super().__init__()
        position = jnp.arange(max_seq_len).reshape(-1, 1)
        div_term = jnp.exp(
            jnp.arange(0, token_dim, 2) * (-jnp.log(10000.0) / token_dim)
        )
        pe = jnp.zeros((1, max_seq_len, token_dim))
        pe = pe.at[..., 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[..., 1::2].set(jnp.cos(position * div_term))
        self.pe = pe

    def __call__(self, x: Array, rng=None) -> Array:
        """
        Arguments:
            x: jnp.ndarray, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:, : x.shape[1]]
        return x


class LearnedPosEmbed(hk.Module):
    def __init__(self, max_seq_len: int, name: str = "learned_pos_embed"):
        super().__init__(name=name)
        self.max_seq_len = max_seq_len
        self.embed_init = hk.initializers.TruncatedNormal(stddev=0.02)

    def __call__(self, x: Array, rng=None) -> Array:
        """Embeds the input with learned positional embeddings.

        Args:
            x (Array): Input array of shape [B, T, D]
            max_len (int, optional): Maximum length of the sequence. Defaults to 512.

        Returns:
            Array: Output array of shape [B, T, D]
        """
        _, seq_len, embed_dim = x.shape
        assert (
            seq_len <= self.max_seq_len
        ), "Sequence length cannot be greater than max_len"
        positional_embeddings = hk.get_parameter(
            "positional_embeddings", [self.max_seq_len, embed_dim], init=self.embed_init
        )
        positional_embeddings = positional_embeddings[:seq_len, :]
        return x + positional_embeddings[None, :, :]
