

from abc import ABC, abstractmethod
from functools import partial

import jax 
import jax.numpy as jnp



class Task(ABC):
    
    def __init__(self, name: str, backend: str = "jax") -> None:
        self.name = name
        self.backend = backend    
        
    @property
    def theta_dim(self):
        return self.get_theta_dim()
    
    @property
    def x_dim(self):
        return self.get_x_dim()
    
    def get_theta_dim(self):
        raise NotImplementedError()
    
    def get_x_dim(self):
        raise NotImplementedError()
    
    def get_data(self, num_samples: int, rng=None):
        raise NotImplementedError()
    
    def get_node_id(self):
        raise NotImplementedError()
    
    def get_batch_sampler(self):
        return base_batch_sampler
    

    def get_base_mask_fn(self):
        raise NotImplementedError()
    
    
class InferenceTask(Task):
    
    observations = range(1, 11)
    
    def __init__(self, name: str, backend: str = "jax") -> None:
        super().__init__(name, backend)
        
    def get_prior(self):
        raise NotImplementedError()
        
    def get_simulator(self):
        raise NotImplementedError()
    
    def get_data(self, num_samples: int, rng=None):
        raise NotImplementedError()
    
    def get_observation(self, index: int):
        raise NotImplementedError()
    
    def get_reference_posterior_samples(self, index: int):
        raise NotImplementedError()
    
    def get_true_parameters(self, index: int):
        raise NotImplementedError()
    
    
class AllConditionalTask(Task):
    
    var_names: list[str]
    
    def __init__(self, name: str, backend: str = "jax") -> None:
        super().__init__(name, backend)
        
    def get_joint_sampler(self):
        raise NotImplementedError()
    
    def get_data(self, num_samples: int, rng=None):
        raise NotImplementedError()
    
    def get_observation_generator(self):
        raise NotImplementedError()
    
    def get_base_mask_fn(self):
        raise NotImplementedError()
        
    def get_reference_sampler(self):
        raise NotImplementedError()

partial(jax.jit, static_argnums=(1, 5))
def base_batch_sampler(key, batch_size, data, node_id, meta_data=None, num_devices=1):
    assert data.ndim == 3, "Data must be 3D, (num_samples, num_nodes, dim)"
    assert (
        node_id.ndim == 2 or node_id.ndim == 1
    ), "Node id must be 2D or 1D, (num_nodes, dim) or (num_nodes,)"

    index = jax.random.randint(key, shape=(num_devices,batch_size,), minval=0, maxval=data.shape[0])
    data_batch = data[index,...]
    node_id_batch = jnp.repeat(node_id[None, ...], num_devices, axis=0).astype(
        jnp.int32
    )
    if meta_data is not None:
        if meta_data.ndim == 3:
            meta_data_batch = meta_data[index,...]
        else:
            meta_data_batch = jnp.repeat(meta_data[None, ...], num_devices, axis=0)
    else:
        meta_data_batch = None
    return data_batch, node_id_batch, meta_data_batch
    
    
    
    