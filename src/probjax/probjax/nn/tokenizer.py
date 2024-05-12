from typing import Optional, Type
import jax
import jax.numpy as jnp

from typing import Callable, Union, Tuple
from jaxtyping import Array, PyTree

import haiku as hk

from probjax.nn.helpers import GaussianFourierEmbedding, SinusoidalEmbedding


def scalarize(data: PyTree) -> PyTree:
    flat, tree = jax.tree_util.tree_flatten(data)
    flat = [jnp.expand_dims(x, -1) for x in flat]
    flat_concat = jnp.concatenate(flat, axis=-2)

    tree_type, tree_data = tree.node_data()
    if tree_type is dict:
        node_num = flat_concat.shape[-2]
        node_names_num = len(tree_data)
        keys = [
            n + "_" + str(i)
            for n in tree_data
            for i in range(node_num // node_names_num)
        ]
        values = jnp.split(flat_concat, node_num, axis=-2)
        return dict(zip(keys, values))
    elif tree_type is list:
        return jnp.split(flat_concat, len(tree_data), axis=-2)
    elif tree_type is tuple:
        return tuple(jnp.split(flat_concat, len(tree_data), axis=-2))
    else:
        raise ValueError(f"Unknown tree type: {tree_type}")


class Tokenizer(hk.Module):
    def __init__(
        self,
        output_dim: int,
        node_embeding_builder: Optional[Callable] = None,
        value_embeding_builder: Optional[Callable] = None,
        node_meta_data_embeding_builder: Optional[Callable] = None,
        distributor: Optional[Union[Callable, str]] = "equal",
        accummulator: Optional[Union[Callable, str]] = "concat",
        learn_node_embeding: bool = True,
        learn_value_embeding: bool = False,
        learn_meta_data_embeding: bool = False,
        name: str | None = "tokenizer",
    ):
        """Base class for tokenizers."""
        self.output_dim = output_dim
        self.value_embeding_builder = value_embeding_builder
        self.node_embeding_builder = node_embeding_builder
        self.meta_data_embeding_builder = node_meta_data_embeding_builder
        self.distibutor = distributor
        self.accummulator = accummulator
        self.learn_node_embeding = learn_node_embeding
        self.learn_value_embeding = learn_value_embeding
        self.learn_meta_data_embeding = learn_meta_data_embeding
        super().__init__(name)
        
    @hk.transparent
    def distribute_output_dim(self, with_meta_data: bool = False):
        if isinstance(self.distibutor, Callable):
            return self.distibutor(self.output_dim)
        else:
            if self.accummulator == "concat":
                if with_meta_data:
                    output_dim1 = self.output_dim // 3
                    output_dim2 = self.output_dim // 3
                    output_dim3 = self.output_dim - (output_dim1 + output_dim2)
                    return output_dim1, output_dim2, output_dim3
                else:
                    output_dim1 = self.output_dim // 2
                    output_dim2 = self.output_dim - output_dim1
                    return output_dim1, output_dim2, 0
            elif self.accummulator == "sum":
                return self.output_dim, self.output_dim, self.output_dim
            else:
                raise ValueError(
                    f"Unknown accummulator: {self.accummulator}, Please specify a custom distributor function, that returns a tuple of output dimensions for each input type."
                )
                
    @hk.transparent
    def accumulate(self, data_id_embedding, data_embedding, meta_data_embedding):
        if self.accummulator == "concat":
            out = [data_id_embedding, data_embedding]
            if meta_data_embedding is not None:
                out.append(meta_data_embedding)
            return jnp.concatenate(out, axis=-1)
        elif self.accummulator == "sum":
            out = data_id_embedding + data_embedding
            if meta_data_embedding is not None:
                out += meta_data_embedding
            return out

        else:
            raise ValueError(
                f"Unknown accummulator: {self.accummulator}, Please specify a custom distributor function, that returns a tuple of output dimensions for each input type."
            )


class ScalarTokenizer(Tokenizer):
    def __init__(
        self,
        output_dim: int,
        max_sequence_length: int,
        node_embeding_builder: Optional[Callable] = None,
        value_embeding_builder: Optional[Callable] = None,
        node_meta_data_embeding_builder: Optional[Callable] = None,
        accummulator: Optional[Union[Callable, str]] = "concat",
        distributor: Optional[Callable] = None,
        learn_node_embeding: bool = True,
        learn_value_embeding: bool = False,
        learn_meta_data_embeding: bool = True,
        name: str | None = "scalar_tokenizer",
    ):
        """Tokenize a scalar data into a vector, by concatenating the node id, value and additional meta data.

        Args:
            output_dim (int): Output dimension of the tokenized data.
            max_sequence_length (int): Maximum number of nodes in the graph (used for node id embeding).
            node_embeding_builder (Optional[Callable], optional): Should be a function f: (id (int32), out_dim1 (int)) -> embeding (float32) . Defaults to None. Which uses a learnable embedding vector for each node.
            value_embeding_builder (Optional[Callable], optional): Should get a function f: (val (float32), out_dim2 (int)) -> embeding (float32). Defaults to None. Which will duplicate the value of each node out_dim2 times.
            node_meta_data_embeding_builder (Optional[Callable], optional): Should get a function f: (meta_data (abstract), out_dim3 (int)) -> embeding (float32). Defaults to None. Which will use a Gaussian Fourier Embedding for each meta data.
            accummulator (Optional[Union[Callable, str]], optional): Either a string ("concat" or "add") or a function f:(embed_id, embed_val, embeding_meta) -> embedding. Defaults to "concat".
            distributor (Optional[Callable], optional): A function f: out_dim (int) -> (out_dim1, out_dim2, out_dim3). Defaults to None.
            name (str | None, optional): Name of the module. Defaults to "scalar_tokenizer".
        """
        self.max_sequence_length = max_sequence_length
        super().__init__(
            output_dim,
            node_embeding_builder,
            value_embeding_builder,
            node_meta_data_embeding_builder,
            distributor,
            accummulator,
            learn_node_embeding,
            learn_value_embeding,
            learn_meta_data_embeding,
            name,
        )

    def __call__(self, data_id: Array, data: Array, meta_data: Optional[Array] = None):
        *leading_dims, sequence_length, variable_dim = data.shape

        data = data.reshape(-1, sequence_length, variable_dim)
        data_id = data_id.astype(jnp.int32).reshape(-1, sequence_length, 1)
        data_id, data = jnp.broadcast_arrays(data_id, data)

        if meta_data is not None:
            meta_data.reshape(-1, sequence_length, variable_dim)
            data_id, data, meta_data = jnp.broadcast_arrays(data_id, data, meta_data)

        output_dim1, output_dim2, output_dim3 = self.distribute_output_dim(
            with_meta_data=meta_data is not None
        )

        data_id_embeding = self.node_embeding(data_id, output_dim1)
        data_embeding = self.value_embeding(data, output_dim2)

        if meta_data is not None:
            meta_data_embeding = self.meta_data_embeding(meta_data, output_dim3)
        else:
            meta_data_embeding = None

        tokens = self.accumulate(data_id_embeding, data_embeding, meta_data_embeding)

        return tokens.reshape(*leading_dims, sequence_length, self.output_dim)


    @hk.transparent
    def value_embeding(self, value, output_dim):
        if self.value_embeding_builder is None:
            value_embeding_fn = lambda x: jnp.repeat(x, output_dim, axis=-1)
        else:
            value_embeding_fn = self.value_embeding_builder(output_dim)

        out = value_embeding_fn(value).reshape(-1, value.shape[-2], output_dim)
        if self.learn_value_embeding:
            out = jax.lax.stop_gradient(out)
        return out

    @hk.transparent
    def node_embeding(self, node, output_dim):
        if self.node_embeding_builder is None:
            node_embeding_fn = hk.Embed(
                self.max_sequence_length,
                output_dim,
                w_init=hk.initializers.Orthogonal(scale=0.5),
            )
        else:
            node_embeding_fn = self.node_embeding_builder(output_dim)

        out = node_embeding_fn(node).reshape(-1, node.shape[-2], output_dim)
        if self.learn_node_embeding:
            out = jax.lax.stop_gradient(out)
        return out

    @hk.transparent
    def meta_data_embeding(self, meta_data, output_dim):
        if self.meta_data_embeding_builder is None:
            meta_data_embeding_fn = hk.Sequential([GaussianFourierEmbedding(256), hk.Linear(output_dim)])
        else:
            meta_data_embeding_fn = self.meta_data_embeding_builder(output_dim)

        out = meta_data_embeding_fn(meta_data).reshape(
            -1, meta_data.shape[-2], output_dim
        )
        out = jnp.nan_to_num(out, nan=0.0)
        if self.learn_meta_data_embeding:
            out = jax.lax.stop_gradient(out)
        return out
    
    
def value_embeding_functions(output_dim, max_sequence_length):
    def f(index,x):
        experts = [hk.Linear(output_dim) for _ in range(max_sequence_length)]
        if hk.running_init():
        # During init unconditionally create params/state for all experts.
            for expert in experts:
                out = expert(x)
        else:
            # During apply conditionally apply (and update) only one expert.
            out = hk.switch(index, experts, x)
        return out
    
    init_fn, apply_fn = hk.without_apply_rng(hk.transform(f))
    return init_fn, apply_fn



class StructuredTokenizer(Tokenizer):
    def __init__(
        self,
        output_dim: int,
        max_sequence_length: int,
        data_name_to_id: dict[str, int],
        value_embeding_builder: Optional[Callable] = None,
        node_embeding_builder: Optional[Callable] = None,
        node_meta_data_embeding_builder: Optional[Callable] = None,
        distributor: Optional[Union[Callable, str]] = "equal",
        accummulator: Optional[Union[Callable, str]] = "concat",
        learn_node_embeding: bool = True,
        learn_value_embeding: bool = False,
        learn_meta_data_embeding: bool = False,
        name: str | None = "tokenizer",
    ):
        self.max_sequence_length = max_sequence_length
        self.data_name_to_id = data_name_to_id
        super().__init__(
            output_dim,
            node_embeding_builder,
            value_embeding_builder,
            node_meta_data_embeding_builder,
            distributor,
            accummulator,
            learn_node_embeding,
            learn_value_embeding,
            learn_meta_data_embeding,
            name,
        )

    def __call__(self, data: dict[str,Array], meta_data: Optional[PyTree] = None):
        output_dim1, output_dim2, output_dim3 = self.distribute_output_dim(
            with_meta_data=meta_data is not None
        )
        
        data_id = jnp.array([self.data_name_to_id[k] for k in data.keys()])
        data_id_embeding = self.node_embeding(data_id, output_dim1)
        value_embeding = self.value_embeding(data, output_dim2)
        if meta_data is not None:
            meta_data_embeding = self.meta_data_embeding(meta_data, output_dim3)
        else:
            meta_data_embeding = None
        
        if meta_data_embeding is  None:
            data_id_embeding, value_embeding = jnp.broadcast_arrays(
                data_id_embeding, value_embeding
            )
        else:
            data_id_embeding, value_embeding, meta_data_embeding = jnp.broadcast_arrays(
                data_id_embeding, value_embeding, meta_data_embeding
            )
        
        tokens = self.accumulate(data_id_embeding, value_embeding, meta_data_embeding)
        
        return tokens
        
        
    @hk.transparent
    def value_embeding(self, value, output_dim):
        if self.value_embeding_builder is None:
            value_embeding_fns = {}
            for k in value:
                value_embeding_fns[k] = hk.Linear(
                    output_dim,
                )
        else:
            value_embeding_fns = self.value_embeding_builder(id, value, output_dim)

        tokens = []
        for k in value:
            out = value_embeding_fns[k](value[k])
            if self.learn_value_embeding:
                out = jax.lax.stop_gradient(out)
            tokens.append(out[:, None, :])
            
        tokens = jnp.concatenate(tokens, axis=-2)
        return tokens

    @hk.transparent
    def node_embeding(self, node, output_dim):
        if self.node_embeding_builder is None:
            node_embeding_fn = hk.Embed(
                self.max_sequence_length,
                output_dim,
                w_init=hk.initializers.Orthogonal(scale=0.5),
            )
        else:
            node_embeding_fn = self.node_embeding_builder(output_dim)

        out = node_embeding_fn(node)
        if self.learn_node_embeding:
            out = jax.lax.stop_gradient(out)
        return out

