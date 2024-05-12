import jax
import jax.numpy as jnp

import haiku as hk

from typing import Callable, Optional

from probjax.nn.helpers import GaussianFourierEmbedding
from probjax.nn.tokenizer import ScalarTokenizer, StructuredTokenizer
from probjax.nn.transformers import Transformer


def conditional_mlp(
    output_dim: int,
    hidden_dim: int = 100,
    num_hidden: int = 8,
    activation=jax.nn.gelu,
    layer_norm: bool = True,
    output_scale_fn=None,
    **kwargs,
):
    """Just builds a conditional score model with MLPs. As the score is typically grows proportional to the variance of the marginal sde, it is useful to scale the output."""

    if output_scale_fn is None:
        output_scale_fn = lambda t, x: x

    def score_net(t, x, context):
        x = jnp.concatenate([x, context], axis=-1)

        time_embedding = GaussianFourierEmbedding(hidden_dim)(t[..., None])
        h = activation(hk.Linear(hidden_dim)(x) + time_embedding)

        for _ in range(num_hidden - 1):
            h_new = hk.Linear(hidden_dim)(h)
            h_new += time_embedding
            h = activation(h_new)

            if layer_norm:
                h = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(h)

        out = hk.Linear(output_dim)(h)
        out = output_scale_fn(t, out)
        return out

    init_fn, apply_fn = hk.without_apply_rng(hk.transform(score_net))
    return init_fn, apply_fn


def scalar_transformer_model(
    num_nodes: int,
    token_dim: int =40,
    condition_token_dim: int =10,
    condition_token_init_scale: int =0.01,
    condition_token_init_mean: int=0.0,
    condition_mode: str="concat",
    time_embedding_dim: int=128,
    num_heads: int=8,
    num_layers: int=6,
    attn_size: int=5,
    widening_factor: int=4,
    num_hidden_layers: int=1,
    act=jax.nn.gelu,
    skip_connection_attn: bool=True,
    skip_connection_mlp: bool=True,
    layer_norm: bool=True,
    output_scale_fn=None,
    base_mask=None,
    **kwargs,
):
    if output_scale_fn is None:
        output_scale_fn = lambda t, x: x

    if condition_mode == "concat":
        condition_token_dim = condition_token_dim
    elif condition_mode == "add":
        token_dim = token_dim + condition_token_dim
        condition_token_dim = token_dim
    elif condition_mode == "none":
        token_dim = token_dim + condition_token_dim
        condition_token_dim = 0

    def model(t, data, data_id, condition_mask, meta_data=None, edge_mask=base_mask):
        _, current_nodes, _ = data.shape  # (batch, nodes, 1)
        data_id = data_id.reshape(-1, current_nodes)
        condition_mask = condition_mask.reshape(-1, current_nodes)

        tokenizer = ScalarTokenizer(token_dim, num_nodes)
        time_embeder = GaussianFourierEmbedding(time_embedding_dim)

        # Embedding
        tokens = tokenizer(data_id, data, meta_data)
        time = time_embeder(t[..., None])

        # Conditioning
        if condition_mode != "none":
            condition_token = hk.get_parameter(
                "condition_token",
                shape=[1, 1, condition_token_dim],
                init=hk.initializers.RandomNormal(
                    condition_token_init_scale, condition_token_init_mean
                ),
            )
            condition_mask = condition_mask.reshape(-1, current_nodes, 1)
            condition_token = condition_mask * condition_token
            if condition_mode == "add":
                tokens = tokens + condition_token
            elif condition_mode == "concat":
                condition_token = jnp.broadcast_to(
                    condition_token, tokens.shape[:-1] + (condition_token_dim,)
                )
                tokens = jnp.concatenate([tokens, condition_token], -1)

        # Forward pass

        model = Transformer(
            num_heads=num_heads,
            num_layers=num_layers,
            attn_size=attn_size,
            widening_factor=widening_factor,
            num_hidden_layers=num_hidden_layers,
            act=act,
            skip_connection_attn=skip_connection_attn,
            skip_connection_mlp=skip_connection_mlp,
 #           layer_norm=layer_norm,
        )

        h = model(tokens, context=time, mask=edge_mask)
        out = hk.Linear(1)(h)
        out = output_scale_fn(t, out)
        return out

    init_fn, model_fn = hk.without_apply_rng(hk.transform(model))
    return init_fn, model_fn


def structured_transformer_model(
    num_nodes: int,
    data_name_to_id: dict,
    token_dim: int =40,
    condition_token_dim: int =10,
    condition_token_init_scale: int =0.01,
    condition_token_init_mean: int=0.0,
    condition_mode: str="concat",
    time_embedding_dim: int=128,
    num_heads: int=4,
    num_layers: int=6,
    attn_size: int=5,
    widening_factor: int=4,
    num_hidden_layers: int=1,
    act=jax.nn.gelu,
    value_embeding_builder: Optional[Callable] = None,
    node_embeding_builder: Optional[Callable] = None,
    node_meta_data_embeding_builder: Optional[Callable] = None,
    skip_connection_attn: bool=True,
    skip_connection_mlp: bool=True,
    layer_norm: bool=True,
    output_scale_fn=None,
    base_mask=None,
    **kwargs,
):
    if output_scale_fn is None:
        output_scale_fn = lambda t, x: x

    if condition_mode == "concat":
        condition_token_dim = condition_token_dim
    elif condition_mode == "add":
        token_dim = token_dim + condition_token_dim
        condition_token_dim = token_dim
    elif condition_mode == "none":
        token_dim = token_dim + condition_token_dim
        condition_token_dim = 0

    def model(t, data, condition_mask, meta_data=None, edge_mask=base_mask):
        current_nodes = len(data)
        data_dim = jax.tree_map(lambda x: x.shape[-1], data)
        condition_mask = condition_mask.reshape(-1, current_nodes)
        

        tokenizer = StructuredTokenizer(token_dim, num_nodes, data_name_to_id,value_embeding_builder, node_embeding_builder, node_meta_data_embeding_builder)
        time_embeder = GaussianFourierEmbedding(time_embedding_dim)
        output_layers = jax.tree_map(lambda x: hk.Linear(x), data_dim)

        # Embedding
        tokens = tokenizer(data, meta_data)
        time = time_embeder(t[..., None])

        # Conditioning
        if condition_mode != "none":
            condition_token = hk.get_parameter(
                "condition_token",
                shape=[1, 1, condition_token_dim],
                init=hk.initializers.RandomNormal(
                    condition_token_init_scale, condition_token_init_mean
                ),
            )
            condition_mask = condition_mask.reshape(-1, current_nodes, 1)
            condition_token = condition_mask * condition_token
            if condition_mode == "add":
                tokens = tokens + condition_token
            elif condition_mode == "concat":
                condition_token = jnp.broadcast_to(
                    condition_token, tokens.shape[:-1] + (condition_token_dim,)
                )
                tokens = jnp.concatenate([tokens, condition_token], -1)

        # Forward pass

        model = Transformer(
            num_heads=num_heads,
            num_layers=num_layers,
            attn_size=attn_size,
            widening_factor=widening_factor,
            num_hidden_layers=num_hidden_layers,
            act=act,
            skip_connection_attn=skip_connection_attn,
            skip_connection_mlp=skip_connection_mlp,
 #           layer_norm=layer_norm,
        )

        h = model(tokens, context=time, mask=edge_mask)
        outs = []
        for i,key in enumerate(data):
            h_i = h[..., i, :]
            out = output_layers[key](h_i)
            outs.append(out)
        out = jnp.concatenate(outs,axis=-1)
        out = output_scale_fn(t, out)
        return out

    init_fn, model_fn = hk.without_apply_rng(hk.transform(model))
    return init_fn, model_fn