from sbibm import get_task as _get_torch_task

import torch
import jax
import jax.numpy as jnp

from scoresbibm.tasks.base_task import InferenceTask


class SBIBMTask(InferenceTask):
    observations = range(1, 11)

    def __init__(self, name: str, backend: str = "jax") -> None:
        super().__init__(name, backend)
        self.task = _get_torch_task(self.name)
        
    def get_theta_dim(self):
        return self.task.dim_parameters
    
    def get_x_dim(self):
        return self.task.dim_data

    def get_prior(self):
        if self.backend == "torch":
            return self.task.get_prior_dist()
        else:
            raise NotImplementedError()

    def get_simulator(self):
        if self.backend == "torch":
            return self.task.get_simulator()
        else:
            raise NotImplementedError()
    
    def get_node_id(self):
        dim = self.get_theta_dim() + self.get_x_dim()
        if self.backend == "torch":
            return torch.arange(dim)
        else:
            return jnp.arange(dim)

    def get_data(self, num_samples: int, **kwargs):
        try:
            prior = self.get_prior()
            simulator = self.get_simulator()
            thetas = prior.sample((num_samples,))
            xs = simulator(thetas)
            return {"theta":thetas, "x":xs}
        except:
            # If not implemented in JAX, use PyTorch
            old_backed = self.backend
            self.backend = "torch"
            prior = self.get_prior()
            simulator = self.get_simulator()
            thetas = prior.sample((num_samples,))
            xs = simulator(thetas)
            self.backend = old_backed
            if self.backend == "numpy":
                thetas = thetas.numpy()
                xs = xs.numpy()
            elif self.backend == "jax":
                thetas = jnp.array(thetas)
                xs = jnp.array(xs)
            return {"theta":thetas, "x":xs}

    def get_observation(self, index: int):
        if self.backend == "torch":
            return self.task.get_observation(index)
        else:
            out = self.task.get_observation(index)
            if self.backend == "numpy":
                return out.numpy()
            elif self.backend == "jax":
                return jnp.array(out)

    def get_reference_posterior_samples(self, index: int):
        if self.backend == "torch":
            return self.task.get_reference_posterior_samples(index)
        else:
            out = self.task.get_reference_posterior_samples(index)
            if self.backend == "numpy":
                return out.numpy()
            elif self.backend == "jax":
                return jnp.array(out)

    def get_true_parameters(self, index: int):
        if self.backend == "torch":
            return self.task.get_true_parameters(index)
        else:
            out = self.task.get_true_parameters(index)
            if self.backend == "numpy":
                return out.numpy()
            elif self.backend == "jax":
                return jnp.array(out)



class LinearGaussian(SBIBMTask):
    def __init__(self, backend: str = "torch") -> None:
        super().__init__(name="gaussian_linear", backend=backend)
        
    def get_base_mask_fn(self):
        task = _get_torch_task(self.name)
        theta_dim = task.dim_parameters
        x_dim = task.dim_data
        thetas_mask = jnp.eye(theta_dim, dtype=jnp.bool_)
        x_i_mask = jnp.eye(x_dim, dtype=jnp.bool_)
        base_mask = jnp.block([[thetas_mask, jnp.zeros((theta_dim, x_dim))], [jnp.eye((x_dim)), x_i_mask]])
        base_mask = base_mask.astype(jnp.bool_)
        
        def base_mask_fn(node_ids, node_meta_data):
            return base_mask[node_ids, :][:, node_ids]
        
        return base_mask_fn
    
    
class BernoulliGLM(SBIBMTask):
    def __init__(self, backend: str = "torch") -> None:
        super().__init__(name="bernoulli_glm", backend=backend)
        
    def get_base_mask_fn(self):
        raise NotImplementedError()
    

class BernoulliGLMRaw(SBIBMTask):
    def __init__(self, backend: str = "torch") -> None:
        super().__init__(name="bernoulli_glm_raw", backend=backend)
        
    def get_base_mask_fn(self):
        raise NotImplementedError()
    



class MixtureGaussian(SBIBMTask):
    def __init__(self, backend: str = "torch") -> None:
        super().__init__(name="gaussian_mixture", backend=backend)
        
    def get_base_mask_fn(self):
        task = _get_torch_task(self.name)
        theta_dim = task.dim_parameters
        x_dim = task.dim_data
        thetas_mask = jnp.eye(theta_dim, dtype=jnp.bool_)
        x_mask = jnp.tril(jnp.ones((theta_dim, x_dim), dtype=jnp.bool_))
        base_mask = jnp.block([[thetas_mask, jnp.zeros((theta_dim, x_dim))], [jnp.ones((x_dim, theta_dim)), x_mask]])
        base_mask = base_mask.astype(jnp.bool_)

        def base_mask_fn(node_ids, node_meta_data):
            return base_mask[node_ids, :][:, node_ids]
        
        return base_mask_fn
        
    

class TwoMoons(SBIBMTask):
    def __init__(self, backend: str = "torch") -> None:
        super().__init__(name="two_moons", backend=backend)
        
    def get_base_mask_fn(self):
        task = _get_torch_task(self.name)
        theta_dim = task.dim_parameters
        x_dim = task.dim_data
        thetas_mask = jnp.eye(theta_dim, dtype=jnp.bool_)
        x_mask = jnp.tril(jnp.ones((theta_dim, x_dim), dtype=jnp.bool_))
        base_mask = jnp.block([[thetas_mask, jnp.zeros((theta_dim, x_dim))], [jnp.ones((x_dim, theta_dim)), x_mask]])
        base_mask = base_mask.astype(jnp.bool_)
        def base_mask_fn(node_ids, node_meta_data):
            return base_mask[node_ids, :][:, node_ids]
        
        return base_mask_fn
        

class SLCP(SBIBMTask):
    def __init__(self, backend: str = "torch") -> None:
        super().__init__(name="slcp", backend=backend)
        
    def get_base_mask_fn(self):
        task = _get_torch_task(self.name)
        theta_dim = task.dim_parameters
        x_dim = task.dim_data
        thetas_mask = jnp.eye(theta_dim, dtype=jnp.bool_) 
        # TODO This could be triangular -> DAG
        x_i_dim = x_dim // 4
        x_i_mask = jax.scipy.linalg.block_diag(*tuple([jnp.tril(jnp.ones((x_i_dim,x_i_dim), dtype=jnp.bool_))]*4)) 
        base_mask = jnp.block([[thetas_mask, jnp.zeros((theta_dim,x_dim))], [jnp.ones((x_dim, theta_dim)), x_i_mask]]) 
        base_mask = base_mask.astype(jnp.bool_)
        def base_mask_fn(node_ids, node_meta_data):
            # If node_ids are permuted, we need to permute the base_mask
            return base_mask[node_ids, :][:, node_ids]
        
        return base_mask_fn
