import functools
import math

import jax
import jax.numpy as jnp
import numpy as np

import haiku as hk
from typing import Callable, Sequence, Optional, Union, Any, Tuple, Iterable

import warnings


class MultiHeadAttention(hk.MultiHeadAttention):
    def __init__(
        self,
        *args,
        save_attention_weights: bool = False,
        attention_method="dense",
        **kwargs,
    ):
        self.save_attention_weights = save_attention_weights
        self.attention_method = attention_method

        super().__init__(*args, **kwargs)

    def __call__(
        self,
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
        mask: Optional[jax.Array] = None,
    ) -> jax.Array:
        # In shape hints below, we suppress the leading dims [...] for brevity.
        # Hence e.g. [A, B] should be read in every case as [..., A, B].
        projection = self._linear_projection

        # Compute key/query/values (overload K/Q/V to denote the respective sizes).
        query_heads = projection(query, self.key_size, "query")  # [T', H, Q=K]
        key_heads = projection(key, self.key_size, "key")  # [T, H, K]
        value_heads = projection(value, self.value_size, "value")  # [T, H, V]

        if self.attention_method == "dense":
            attn, attn_weights = dense_dot_product_attention(
                query_heads,
                key_heads,
                value_heads,
                self.key_size,
                mask,
                self.save_attention_weights,
            )
        elif self.attention_method == "mem_eff":
            attn = memory_efficient_dot_product_attention(
                query_heads,
                key_heads,
                value_heads,
                mask,
                self.save_attention_weights,
            )
            attn_weights = None
            return attn
        elif self.attention_method == "sparse":
            attn = sparse_dot_product_attention(
                query_heads,
                key_heads,
                value_heads,
                mask,
                self.save_attention_weights,
            )
            attn_weights = None
            return attn
        else:
            raise NotImplementedError("Unimplemented attention method")

        if self.save_attention_weights:
            _ = hk.get_state(
                "attn_weights",
                shape=attn_weights.shape,
                dtype=attn_weights.dtype,
                init=hk.initializers.Constant(0.0),
            )
            hk.set_state("attn_weights", attn_weights)

        # Apply another projection to get the final embeddings.
        final_projection = hk.Linear(
            self.model_size,
            w_init=self.w_init,
            with_bias=self.with_bias,
            b_init=self.b_init
        )
        return final_projection(attn)  # [T', D']


def dense_dot_product_attention(
    query_heads,  # [...,T', H, K]
    key_heads,  # [...,T', H, K]
    value_heads,  # [T, H, V]
    key_size: int,
    mask=None,  # [..., T,T]
    return_attention_weights: bool = False,
):
    *leading_dims, sequence_length, _, _ = query_heads.shape
    attn_logits = jnp.einsum("...thd,...Thd->...htT", query_heads, key_heads)
    attn_logits = attn_logits / np.sqrt(key_size).astype(key_heads.dtype)

    if mask is not None:
        if mask.ndim != attn_logits.ndim:
            raise ValueError(
                f"Mask dimensionality {mask.ndim} must match logits dimensionality "
                f"{attn_logits.ndim}."
            )
        attn_logits = jnp.where(mask, attn_logits, -1e30)
    attn_weights = jax.nn.softmax(attn_logits)  # [H, T', T]

    attn = jnp.einsum("...htT,...Thd->...thd", attn_weights, value_heads)
    attn = jnp.reshape(attn, (*leading_dims, sequence_length, -1))  # [T', H*V]

    if return_attention_weights:
        return attn, attn_weights
    else:
        return attn, None


def memory_efficient_dot_product_attention(
    query,
    key,
    value,
    mask=None,
    precision=jax.lax.Precision.HIGHEST,
    query_chunk_size=512,
    key_chunk_size=2048,
):
    """Computes efficient dot-product attention given query, key, and value.
    
    Args:
        query: The query tensor of shape (..., num_q, num_heads, q_features).
        key: The key tensor of shape (..., num_k, num_heads, k_features).
        value: The value tensor of shape (..., num_k, num_heads, v_features).
        mask: Optional mask tensor of shape (..., num_q, num_k) or (..., num_q, 1).
        precision: The precision level for computation. Defaults to jax.lax.Precision.HIGHEST.
        query_chunk_size: The chunk size for query tensor. Defaults to 512.
        key_chunk_size: The chunk size for key tensor. Defaults to 2048.
    
    Returns:
        The attention output tensor of shape (..., num_q, -1).
    """
    *leading_dims, num_q, num_heads, q_features = query.shape

    query_chunk_size = num_q // math.gcd(num_q, query_chunk_size)

    def chunk_scanner(chunk_idx, _):
        query_chunk = jax.lax.dynamic_slice(
            query,
            tuple([0] * (query.ndim - 3)) + (chunk_idx, 0, 0),
            slice_sizes=tuple(leading_dims)
            + (min(query_chunk_size, num_q), num_heads, q_features),
        )

        if mask is None:
            mask_chunk = None
        elif mask.shape[-2] == 1:
            mask_chunk = mask
        elif mask.shape[-2] == num_q:
            mask_chunk = jax.lax.dynamic_slice(
                mask,
                tuple([0] * (mask.ndim - 3)) + (0, chunk_idx, 0),
                slice_sizes=tuple(leading_dims)
                + (mask.shape[-3], min(query_chunk_size, num_q), mask.shape[-1]),
            )
        else:
            raise TypeError(
                f"mask.shape[-2] == {mask.shape[-2]} must broadcast with query.shape[-3] == {num_q}"
            )

        return (
            chunk_idx + query_chunk_size,
            _query_chunk_attention(
                chunk_idx,
                query_chunk,
                key,
                value,
                mask_chunk,
                precision=precision,
                key_chunk_size=key_chunk_size,
            ),
        )


    l = num_q // query_chunk_size
    _, res = jax.lax.scan(chunk_scanner, init=0, xs=None, length=l)

    res = jnp.concatenate(res, axis=-3)
    res = jnp.reshape(res, (*leading_dims, num_q, -1))
    return res



def sparse_dot_product_attention(
    query_heads,  # [...,T', H, K]
    key_heads,  # [...,T', H, K]
    value_heads,  # [T, H, V]
    indices1,  # Should be the indices where the mask is true
    indices2,
    return_attention_weights: bool = False,
):
    *leading_dims, sequence_length, _, dim = query_heads.shape
    query_heads = jnp.take(
        query_heads, indices1, axis=-3
    )  # [..., E, H, K] Where E is the number of edges
    key_heads = jnp.take(key_heads, indices2, axis=-3)  # [..., E, H, K]
    value_heads = jnp.take(value_heads, indices2, axis=-3)  # [..., E, H, V]

    # Attention logits
    attention_logits = jnp.einsum(
        "...ehd,...ehd->...eh", query_heads, key_heads
    ) / jnp.sqrt(dim).astype(key_heads.dtype)
    attention_logits = attention_logits - jnp.max(
        attention_logits, axis=-2, keepdims=True
    )
    attention_weight = jnp.exp(attention_logits)
    attention_normalizer = jax.ops.segment_sum(
        attention_weight,
        indices1,
        num_segments=sequence_length,
        indices_are_sorted=True,
    )
    attention_normalizer = jnp.take(attention_normalizer, indices1, axis=-2)
    attention_weight = attention_weight / attention_normalizer  # [..., eh]

    # Attention weighted values
    attn = attention_weight[..., None] * value_heads
    attn = jax.ops.segment_sum(
        attn, indices1, num_segments=sequence_length, indices_are_sorted=True
    )
    attn = jnp.reshape(attn, (*leading_dims, sequence_length, -1))  # [T', H*V]

    if return_attention_weights:
        return attn, attention_weight
    else:
        return attn, None

   

def _query_chunk_attention(
    query_idx,
    query,
    key,
    value,
    mask,
    precision,
    key_chunk_size=2048,
):
    num_kv, num_heads, k_features = key.shape[-3:]
    v_features = value.shape[-1]

    key_chunk_size = min(key_chunk_size, num_kv)
    query = query / jnp.sqrt(k_features)
    
    # NOTE: num_kv must be divisible by key_chunk_size
    key_chunk_size = num_kv // math.gcd(num_kv, key_chunk_size)

    @functools.partial(jax.checkpoint, prevent_cse=False)
    def summarize_chunk(chunk_idx, query, key, value, mask):
        attn_weights = jnp.einsum(
            "...qhd,...khd->...qhk", query, key, precision=precision
        )

        max_score = jnp.max(attn_weights, axis=-1, keepdims=True)
        max_score = jax.lax.stop_gradient(max_score)
        exp_weights = jnp.exp(attn_weights - max_score)
        exp_values = jnp.einsum(
            "...vhf,...qhv->...qhf", value, exp_weights, precision=precision
        )
        max_score = jnp.squeeze(max_score, axis=-1)
        return exp_values, exp_weights.sum(axis=-1), max_score

    def chunk_scanner(chunk_idx):
        key_chunk = jax.lax.dynamic_slice(
            key,
            tuple([0] * (key.ndim - 3)) + (chunk_idx, 0, 0),
            slice_sizes=tuple(key.shape[:-3]) + (key_chunk_size, num_heads, k_features),
        )
        value_chunk = jax.lax.dynamic_slice(
            value,
            tuple([0] * (value.ndim - 3)) + (chunk_idx, 0, 0),
            slice_sizes=tuple(value.shape[:-3])
            + (key_chunk_size, num_heads, v_features),
        )

        if mask is None:
            mask_chunk = None
        elif mask.shape[-1] == 1:
            mask_chunk = mask
        elif mask.shape[-1] == num_kv:
            mask_chunk = jax.lax.dynamic_slice(
                mask,
                tuple([0] * (mask.ndim - 3)) + (0, 0, chunk_idx),
                slice_sizes=tuple(mask.shape[:-3])
                + (mask.shape[-3], mask.shape[-2], key_chunk_size),
            )
        else:
            raise TypeError(
                f"mask.shape[-1] == {mask.shape[-1]} must broadcast with key.shape[-3] == {num_kv}"
            )

        return summarize_chunk(
            chunk_idx, query, key_chunk, value_chunk, mask_chunk
        )

    chunk_values, chunk_weights, chunk_max = jax.lax.map(
        chunk_scanner, xs=jnp.arange(0, num_kv, key_chunk_size)
    )


    global_max = jnp.max(chunk_max, axis=0, keepdims=True)
    max_diffs = jnp.exp(chunk_max - global_max)
    chunk_values *= jnp.expand_dims(max_diffs, axis=-1)
    chunk_weights *= max_diffs

    all_values = chunk_values.sum(axis=0)
    all_weights = jnp.expand_dims(chunk_weights, -1).sum(axis=0)
    return all_values / all_weights