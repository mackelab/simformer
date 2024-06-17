from abc import ABC, abstractmethod
from copy import deepcopy

import jax
import jax.numpy as jnp
import torch

from probjax.distributions.continuous import Normal
from probjax.distributions.independent import Independent
from probjax.distributions.transformed_distribution import TransformedDistribution
from probjax.utils.sdeint import sdeint
from probjax.utils.odeint import odeint, _odeint
import numpy as np
import optax

from scoresbibm.methods.guidance import register_classifier_free_guidance, register_generalized_guidance, register_naive_inpaint_guidance, repaint, generalized_guidance, get_constraint_fn

class Model(ABC):
    """
    Abstract base class for models.

    Args:
        method (str): The method used by the model.
        backend (str, optional): The backend used for computation. Defaults to "jax".
    """

    def __init__(self, method: str, backend="jax"):
        self.method = method
        self.backend = backend

    @abstractmethod
    def sample(self, num_samples, x_o, rng=None, **kwargs):
        pass

    @abstractmethod
    def log_prob(self, theta, x_o, **kwargs):
        pass


class PosteriorModel(Model):
    """
    A base class for posterior models.
    """
    
    def set_default_sampling_kwargs(self, **kwargs):
        pass

    def set_default_x_o(self, x_o):
        """
        Set the default value for x_o.

        Args:
            x_o: The default value for x_o.
        """
        self.x_o = x_o

    def _check_x_o(self, x_o):
        """
        Check if x_o is provided, otherwise use the default value.

        Args:
            x_o: The value of x_o.

        Returns:
            The value of x_o.

        Raises:
            ValueError: If x_o is not provided and no default value is set.
        """
        if x_o is None:
            x_o = self.x_o
            if x_o is None:
                raise ValueError(
                    "Please provide x_o, either as argument or by calling set_default_x_o"
                )
        return x_o

    def sample(self, num_samples, x_o=None, rng=None, **kwargs):
        """
        Sample from the posterior model.

        Args:
            num_samples: The number of samples to generate.
            x_o: The value of x_o.
            rng: The random number generator.
            **kwargs: Additional keyword arguments.

        Returns:
            The generated samples.
        """
        x_o = self._check_x_o(x_o)
        return self._sample(num_samples, x_o=x_o, rng=rng, **kwargs)

    def log_prob(self, theta, x_o=None, **kwargs):
        """
        Compute the log probability of theta given x_o.

        Args:
            theta: The value of theta.
            x_o: The value of x_o.
            **kwargs: Additional keyword arguments.

        Returns:
            The log probability.
        """
        x_o = self._check_x_o(x_o)
        return self._log_prob(theta, x_o=x_o, **kwargs)

    @abstractmethod
    def _sample(self, num_samples, x_o, rng=None, **kwargs):
        """
        Abstract method for sampling from the posterior model.

        Args:
            num_samples: The number of samples to generate.
            x_o: The value of x_o.
            rng: The random number generator.
            **kwargs: Additional keyword arguments.
        """
        pass

    @abstractmethod
    def _log_prob(self, theta, x_o, **kwargs):
        """
        Abstract method for computing the log probability of theta given x_o.

        Args:
            theta: The value of theta.
            x_o: The value of x_o.
            **kwargs: Additional keyword arguments.
        """
        pass
    
    def log_prob_batched(self, theta, x_o=None, **kwargs):
        log_probs = []
        for i in range(theta.shape[0]):
            log_probs.append(self.log_prob(theta[i], x_o[i], **kwargs))
            

        return torch.cat(log_probs)


class AllConditionalModel(PosteriorModel):
    """
    A class representing an AllConditionalModel.

    This class extends the PosteriorModel class and provides additional functionality for handling all conditional distributions.

    Args:
        method (str): The method used for modeling.
        backend (str, optional): The backend used for computation. Defaults to "jax".

    Attributes:
        node_id: The default node ID.
        condition_mask: The default condition mask.

    Methods:
        set_default_node_id: Sets the default node ID.
        set_default_condition_mask: Sets the default condition mask.
        _check_id_condition_mask: Checks and retrieves the node ID and condition mask.
        sample: Samples from the model.
        log_prob: Computes the log probability of the model.

    """

    def __init__(self, method: str, backend="jax"):
        super().__init__(method, backend)

        self.node_id = None
        self.condition_mask = None

    def set_default_node_id(self, node_id):
        """
        Sets the default node ID.

        Args:
            node_id: The default node ID.

        """
        self.node_id = node_id

    def set_default_condition_mask(self, condition_mask):
        """
        Sets the default condition mask.

        Args:
            condition_mask: The default condition mask.

        """
        self.condition_mask = condition_mask

    def _check_id_condition_mask(self, node_id, condition_mask):
        """
        Checks and retrieves the node ID and condition mask.

        If the node ID or condition mask is not provided, it retrieves the default values.

        Args:
            node_id: The node ID.
            condition_mask: The condition mask.

        Returns:
            node_id: The node ID.
            condition_mask: The condition mask.

        Raises:
            ValueError: If the node ID or condition mask is not provided.

        """
        if node_id is None:
            node_id = self.node_id
            if node_id is None:
                raise ValueError(
                    "Please provide node_id, either as argument or by calling set_default_node_id"
                )

        if condition_mask is None:
            condition_mask = self.condition_mask
            if condition_mask is not None:
                condition_mask = condition_mask#.astype(jnp.bool_) TODO: Check if this is needed
            else:
                raise ValueError(
                    "Please provide condition_mask, either as argument or by calling set_default_condition_mask"
                )
        return node_id, condition_mask

    def sample(
        self,
        num_samples,
        x_o=None,
        rng=None,
        node_id=None,
        condition_mask=None,
        **kwargs
    ):
        """
        Samples from the model.

        Args:
            num_samples: The number of samples to generate.
            x_o: The observed data.
            rng: The random number generator.
            node_id: The node ID.
            condition_mask: The condition mask.
            **kwargs: Additional keyword arguments.

        Returns:
            The generated samples.

        """
        node_id, condition_mask = self._check_id_condition_mask(node_id, condition_mask)
        return super().sample(
            num_samples,
            x_o,
            rng=rng,
            node_id=node_id,
            condition_mask=condition_mask,
            **kwargs
        )

    def log_prob(self, theta, x_o=None, node_id=None, condition_mask=None, **kwargs):
        """
        Computes the log probability of the model.

        Args:
            theta: The model parameters.
            x_o: The observed data.
            node_id: The node ID.
            condition_mask: The condition mask.
            **kwargs: Additional keyword arguments.

        Returns:
            The log probability.

        """
        node_id, condition_mask = self._check_id_condition_mask(node_id, condition_mask)
        return super().log_prob(
            theta, x_o, node_id=node_id, condition_mask=condition_mask, **kwargs
        )
    
    def log_prob_batched(self, theta, x_o=None, condition_mask=None, **kwargs):
        """
        Computes the log probability of the model for a batch of samples.

        Args:
            theta: The model parameters.
            x_o: The observed data.
            condition_mask: The condition mask.
            num_steps: The number of steps.
            **kwargs: Additional keyword arguments.

        Returns:
            The log probability.

        """
        node_id, condition_mask = self._check_id_condition_mask(None, condition_mask)
        log_prob_fn = lambda theta, x_o: self.log_prob(theta, x_o, node_id=node_id, condition_mask=condition_mask, **kwargs)
        return jax.vmap(log_prob_fn)(theta, x_o)


class SBIPosteriorModel(PosteriorModel):
    def __init__(self, sbi_posterior, method: str):
        super().__init__(method, backend="torch")
        self.sbi_posterior = sbi_posterior

    def _sample(self, num_samples, x_o, rng=None, **kwargs):
        if self.method in ["npe", "nle", "nre"]:
            return self.sbi_posterior.sample((num_samples,), x=x_o)
        else:
            raise NotImplementedError()

    def _log_prob(self, theta, x_o, **kwargs):
        if self.method == "npe":
            return self.sbi_posterior.log_prob(theta, x=x_o)
        elif self.method == "nle":
            return self.sbi_posterior.potential_fn.likelihood_estimator.log_prob(theta, x=x_o)
        else:
            raise NotImplementedError()
        
    def log_prob_batched(self, theta, x_o=None, **kwargs):
        if self.method == "npe":
            if len(theta.shape) == len(x_o.shape):
                return self.sbi_posterior.posterior_estimator.log_prob(theta, context=x_o)
            else:
                log_probs = []
                for i in range(theta.shape[1]):
                    log_probs.append(self.sbi_posterior.posterior_estimator.log_prob(theta[:,i], context=x_o)[:,None])
                return torch.cat(log_probs, dim=1)
        elif self.method == "nle":
            if len(theta.shape) == len(x_o.shape):
                return self.sbi_posterior.potential_fn.likelihood_estimator.log_prob(theta, context=x_o)
            else:
                log_probs = []
                for i in range(theta.shape[1]):
                    log_probs.append(self.sbi_posterior.potential_fn.likelihood_estimator.log_prob(theta[:,i], context=x_o)[:,None])
                return torch.cat(log_probs, dim=1)
        else:
            raise NotImplementedError()
        
    def sample_batched(self, num_samples, x_o, rng=None, **kwargs):
        if self.method in ["npe"]:
            samples = []
            for i in range(x_o.shape[0]):
                samples.append(self.sbi_posterior.sample((num_samples,), x=x_o[i]))
            return torch.stack(samples)
        elif self.method in ["nle", "nre"]:
            samples = []
            for i in range(x_o.shape[0]):
                samples.append(self.sbi_posterior.sample((num_samples,), x=x_o[i]))
            return torch.stack(samples)
        else:   
            raise NotImplementedError()
        



class ScorePosteriorModel(PosteriorModel):
    """A class for the NPSE model. This is a wrapper around the sde and the model. The model is a conditional score model."""

    def __init__(
        self, params, model_fn, sde, model_init_params={}, sde_init_params={}
    ) -> None:
        self.params = params
        self.model_fn = model_fn
        self.sde = sde

        # For sampling
        self.T_min = sde_init_params["T_min"]
        self.T_max = sde_init_params["T_max"]
        self.marginal_end_std = sde.marginal_stddev(jnp.array([self.T_min]))
        self.marginal_end_mean = sde.marginal_mean(jnp.array([self.T_max]))

        # For pickle
        self.model_init_params = model_init_params
        self.sde_init_params = sde_init_params
        self.sampling_kwargs = {"num_steps": 500, "sampling_method": "sde"}

        super().__init__("npse", backend="jax")
        
    def set_default_sampling_kwargs(self, **kwargs):
        self.sampling_kwargs = kwargs

    def _sample(self, num_samples, x_o, num_steps=None, rng=None, **kwargs):
        assert rng is not None, "Please provide a rng key"
        key1, key2 = jax.random.split(rng, 2)
        sampling_kwargs = {**self.sampling_kwargs, **kwargs}
        if num_steps is None:
            num_steps = sampling_kwargs.pop("num_steps")
        
        x_T = (
            jax.random.normal(key1, (num_samples,) + self.sde.event_shape)
            * self.marginal_end_std
            + self.marginal_end_mean
        )
        sampling_method = sampling_kwargs.pop("sampling_method")
        print("Sampling method: ", sampling_method)
        
        if sampling_method == "sde":
            drift, diffusion = self._init_backward_sde(x_o)
            keys = jax.random.split(key2, (num_samples,))
            ys = jax.vmap(
                lambda *args: sdeint(*args, noise_type="diagonal", **sampling_kwargs),
                in_axes=(0, None, None, 0, None),
                out_axes=0,
            )(
                keys,
                drift,
                diffusion,
                x_T,
                jnp.linspace(0.0, self.T_max - self.T_min, num_steps),
            )
            return ys[:, -1, ...]
        elif sampling_method == "ode":
            drift = self._init_backward_ode(x_o)
            ys = jax.vmap(lambda *args:_odeint(*args, **kwargs), in_axes=(None, 0, None))(drift, x_T, jnp.linspace(0.0, self.T_max - self.T_min, num_steps))
            return ys[:, -1, ...]
                

    def _log_prob(self, val, x_o, **kwargs):
        # Add backward ode to compute log_prob
        raise NotImplementedError()

    def _init_backward_sde(self, x_o):
        def drift_backward(t, x):
            t = self.T_max - t
            score = self.model_fn(self.params, jnp.atleast_1d(t), x, jnp.squeeze(x_o))
            drift = self.sde.drift(t, x) - self.sde.diffusion(t, x) ** 2 * score
            return -drift.reshape(x.shape)

        def diffusion_backward(t, x):
            t = self.T_max - t
            return self.sde.diffusion(t, x).reshape(x.shape)

        return drift_backward, diffusion_backward
    
    def _init_backward_ode(self, x_o):
        def drift_backward(t, x):
            t = self.T_max - t
            score = self.model_fn(self.params, jnp.atleast_1d(t), x, jnp.squeeze(x_o))
            dx = self.sde.drift(t,x) - 0.5 * self.sde.diffusion(t, x) ** 2 * score
            return -dx.reshape(x.shape)

        return drift_backward

    def __getstate__(self) -> object:
        state = self.__dict__.copy()
        state["model_fn"] = None
        state["sde"] = None
        return state

    def __setstate__(self, state):
        from scoresbibm.methods.neural_nets import conditional_mlp
        from scoresbibm.methods.sde import init_sde_related
        self.__dict__.update(state)
        self.sde, self.T_min, self.T_max, _, output_scale_fn = init_sde_related(
            **self.sde_init_params
        )
        _, self.model_fn = conditional_mlp(
            output_scale_fn=output_scale_fn, **self.model_init_params
        )


class AllConditionalReferenceModel(AllConditionalModel):
    def __init__(self, sampling_fn, log_prob_fn=None) -> None:
        self.sampling_fn = sampling_fn
        self.log_prob_fn = log_prob_fn
        return super().__init__("reference", backend="jax")

    def _sample(self, num_samples, x_o, rng=None, **kwargs):
        return self.sampling_fn(num_samples, x_o, rng=rng, **kwargs)

    def _log_prob(self, theta, x_o, **kwargs):
        return self.log_prob_fn(theta, x_o, **kwargs)


class AllConditionalScoreModel(AllConditionalModel):
    def __init__(
        self, params, model_fn, sde, sde_init_params, model_init_params, edge_mask_fn_params, z_score_params=None
    ) -> None:
        self.params = params
        self.model_fn = model_fn
        self.sde = sde
        self.edge_mask_fn_params = edge_mask_fn_params
        self.z_score_params = z_score_params

        self.T_min = sde_init_params["T_min"]
        self.T_max = sde_init_params["T_max"]

        self.marginal_end_std = jnp.squeeze(
            sde.marginal_stddev(jnp.array([self.T_max]))
        )
        self.marginal_end_mean = jnp.squeeze(sde.marginal_mean(jnp.array([self.T_max])))

        # For sampling
        self.edge_mask = None
        self.edge_mask_fn = None
        self.meta_data = None
        self.score_fn = self.model_fn  # For score modifcations ...
        self.sampling_kwargs = {"num_steps": 500, "sampling_method": "sde"}

        # For pickle
        self.model_init_params = model_init_params
        self.sde_init_params = sde_init_params

        super().__init__("score_transformer", backend="jax")

    def _check_edge_mask(self, edge_mask, node_id, condition_mask, meta_data):
        if edge_mask is None:
            if self.edge_mask_fn is not None:
                edge_mask = self.edge_mask_fn(node_id, condition_mask, meta_data)
        return edge_mask
    
    def _check_for_meta_data(self, meta_data):
        if meta_data is None:
            if self.meta_data is not None:
                meta_data = self.meta_data
        return meta_data
    
    def _z_score_if_needed(self, x_o, node_id, condition_mask):
        if self.z_score_params is not None:
            z_score_fn = self.z_score_params["z_score_fn"]
            x_o = z_score_fn(x_o, node_id[condition_mask])
        return x_o
    
    def _un_z_score_if_needed(self, theta, node_id, condition_mask):
        if self.z_score_params is not None:
            un_z_score_fn = self.z_score_params["un_z_score_fn"]
            theta = un_z_score_fn(theta, node_id[~condition_mask])
            
        return theta

    def _sample(
        self,
        num_samples,
        x_o,
        num_steps=None,
        node_id=None,
        condition_mask=None,
        meta_data=None,
        edge_mask=None,
        rng=None,
        unique_nodes=False, # If true, we assume all node_ids are unique and are JIT compatible
        return_conditioned_samples=False,  # If true, we preserves shapes and are JIT compatible
        with_bug=False,
        verbose=False,
        **kwargs
    ):
        meta_data = self._check_for_meta_data(meta_data)
        edge_mask = self._check_edge_mask(edge_mask, node_id, condition_mask, meta_data)
        if x_o.shape[0] > 0:
            x_o = self._z_score_if_needed(x_o, node_id, condition_mask)
        
        sampling_kwargs = {**deepcopy(self.sampling_kwargs), **kwargs}
        if num_steps is None:
            num_steps = sampling_kwargs.pop("num_steps")
        else:
            if "num_steps" in sampling_kwargs:
                del sampling_kwargs["num_steps"]
        key1, key2 = jax.random.split(rng, 2)
        if not unique_nodes:
            unique_node_id = jnp.unique(node_id)
            if not with_bug:
                mean_end_per_node = jnp.array([jnp.mean(self.marginal_end_mean[self.node_id ==i]) for i in unique_node_id])
                std_end_per_node = jnp.array([jnp.mean(self.marginal_end_std[self.node_id ==i]) for i in unique_node_id])
            else:
                mean_end_per_node = jnp.array([jnp.mean(self.marginal_end_std[self.node_id ==i]) for i in unique_node_id])
                std_end_per_node = jnp.array([jnp.mean(self.marginal_end_std[self.node_id ==i]) for i in unique_node_id])
        else:
            mean_end_per_node = self.marginal_end_mean
            std_end_per_node = self.marginal_end_std
            
        x_T = (
            jax.random.normal(
                key1,
                (
                    num_samples,
                    node_id.shape[-1],
                ),
            )
            * std_end_per_node[node_id]
            + mean_end_per_node[node_id]
        )
        condition_mask = condition_mask.reshape(x_T.shape[-1])
        
        sampling_method = sampling_kwargs.pop("sampling_method")
        if verbose:
            print("Sampling method: ", sampling_method)
        if sampling_method == "sde":
            if unique_nodes:
                if x_o.shape[0] > 0:
                    indices = jnp.where(condition_mask, jnp.arange(condition_mask.shape[0]), -1) % x_o.shape[0]
                    x_o_pad = x_o[indices]
                    x_T = x_T * (1 - condition_mask) + x_o_pad * condition_mask
            else:
                x_T = x_T.at[..., condition_mask].set(x_o.reshape(-1))
            drift, diffusion = self._init_backward_sde(node_id, condition_mask, edge_mask, meta_data=meta_data)
            keys = jax.random.split(key2, (num_samples,))
            ys = jax.vmap(
                lambda *args: sdeint(*args, noise_type="diagonal",  **sampling_kwargs),
                in_axes=(0, None, None, 0, None),
                out_axes=0,
            )(
                keys,
                drift,
                diffusion,
                x_T,
                jnp.linspace(0.0, self.T_max - self.T_min, num_steps),
            )
            if not return_conditioned_samples:
                final_samples = ys[:, -1, ...][:, ~condition_mask]
            else:
                final_samples = ys[:, -1, ...]
            final_samples = final_samples.reshape((num_samples, -1))

        elif sampling_method == "ode":
            x_T = x_T.at[..., condition_mask].set(x_o.reshape(-1))
            drift = self._init_backward_ode(node_id, condition_mask, edge_mask, meta_data=meta_data)
            ys = jax.vmap(lambda *args:_odeint(*args, **sampling_kwargs), in_axes=(None, 0, None))(drift, x_T, jnp.linspace(0.0, self.T_max - self.T_min, num_steps))
            if not return_conditioned_samples:
                final_samples = ys[:, -1, ...][:, ~condition_mask]
            else:
                final_samples = ys[:, -1, ...]
            final_samples = final_samples.reshape((num_samples, -1))
        elif sampling_method == "repaint":
            resampling_steps = sampling_kwargs.pop("resampling_steps")
            @jax.vmap
            def sample_fn(key, x_T):
                return repaint(self, key, condition_mask, x_o, x_T, num_steps=num_steps, node_id=node_id, edge_mask=edge_mask, meta_data=meta_data, resampling_steps=resampling_steps)
            
            keys = jax.random.split(key2, (num_samples,))
            final_samples = sample_fn(keys, x_T)
            final_samples = final_samples[:,~condition_mask]
        elif sampling_method == "generalized_guidance":
            # Default scaling as inverse marginal variance
            default_scaling_fn_bias = sampling_kwargs.pop("default_scaling_fn_bias", 0.)
            def scaling_fn(t):
                t = jnp.atleast_1d(t)
                std = self.sde.marginal_stddev(t, jnp.array([1.]))
                return (1/(std**2 + default_scaling_fn_bias)) 
            
            scaling_fn = sampling_kwargs.pop("scaling_fn", scaling_fn)
            resampling_steps = sampling_kwargs.pop("resampling_steps",0)
            constraint_name = sampling_kwargs.pop("constraint_name")
            constraints_kwargs = sampling_kwargs.pop("constraint_kwargs", {})
            constraint_mask = sampling_kwargs.pop("constraint_mask", condition_mask)
            condition_mask = condition_mask & ~constraint_mask # If constrained we can't condition on it
            x_T = x_T.at[..., condition_mask].set(x_o.reshape(-1))
            if "constraint_fn" in sampling_kwargs:
                constraint_fn = sampling_kwargs.pop("constraint_fn")
            else:
                constraint_fn = get_constraint_fn(constraint_name, scaling_fn =scaling_fn, constraint_mask=constraint_mask,x_o=x_o, **constraints_kwargs)
            

            @jax.vmap
            def sample_fn(key, x_T):
                return generalized_guidance(self, constraint_fn, key, condition_mask, x_T, num_steps=num_steps, node_id=node_id, edge_mask=edge_mask, meta_data=meta_data, resampling_steps=resampling_steps)
            
            keys = jax.random.split(key2, (num_samples,))
            final_samples = sample_fn(keys, x_T)
            

            if not return_conditioned_samples:
                final_samples = final_samples[:,~condition_mask]
            else:
                final_samples = final_samples
            
        elif sampling_method in ["repaint", "classifier_free_guidance", "naive_inpaint_guidance","generalized_guidance"]:
            if sampling_method == "classifier_free_guidance":
                register_classifier_free_guidance(self, condition_mask, x_o)
                constraint_mask = sampling_kwargs.pop("constraint_mask", condition_mask)
                drift, diffusion = self._init_backward_sde(node_id, jnp.zeros_like(condition_mask), edge_mask, meta_data=meta_data)
            elif sampling_method == "naive_inpaint_guidance":
                register_naive_inpaint_guidance(self, condition_mask, x_o)
                constraint_mask = sampling_kwargs.pop("constraint_mask", condition_mask)
                x_T = x_T.at[..., condition_mask].set(x_o.reshape(-1))
                drift, diffusion = self._init_backward_sde(node_id, condition_mask, edge_mask, meta_data=meta_data)
            # elif sampling_method == "generalized_guidance":
            #     score_manipulator = sampling_kwargs.pop("score_manipulator")
            #     score_manipulator_kwargs = sampling_kwargs.pop("score_manipulator_kwargs", {})
            #     constraint_mask = sampling_kwargs.pop("constraint_mask", condition_mask)
            #     x_T = x_T.at[..., condition_mask & ~constraint_mask].set(x_o.reshape(-1)[:jnp.sum(condition_mask & ~constraint_mask)])
            #     register_generalized_guidance(self, constraint_mask, x_o, score_manipulator=score_manipulator, **score_manipulator_kwargs)
            #     drift, diffusion = self._init_backward_sde(node_id, condition_mask & ~constraint_mask, edge_mask, meta_data=meta_data)
            else:
                raise NotImplementedError()
                
            keys = jax.random.split(key2, (num_samples,))
            ys = jax.vmap(
                lambda *args: sdeint(*args, noise_type="diagonal", **sampling_kwargs),
                in_axes=(0, None, None, 0, None),
                out_axes=0,
            )(
                keys,
                drift,
                diffusion,
                x_T,
                jnp.linspace(0.0, self.T_max - self.T_min, num_steps),
            )
       
            if not return_conditioned_samples:
                final_samples = ys[:, -1, ...][:, ~condition_mask]
            else:
                final_samples = ys[:, -1, ...]
            final_samples = final_samples.reshape((num_samples, -1))
            self.score_fn = self.model_fn
            #return ys
        else:
            raise NotImplementedError()
        
        final_samples = self._un_z_score_if_needed(final_samples, node_id, condition_mask)
        return final_samples

  

    def _log_prob(
            self,
            val,
            x_o,
            num_steps=None,
            node_id=None,
            condition_mask=None,
            meta_data=None,
            edge_mask=None,
            **kwargs,
        ):
        # Add backward ode to compute log_prob
        assert condition_mask is not None, "Please provide a condition mask"
        node_id, condition_mask = self._check_id_condition_mask(node_id, condition_mask)
        meta_data = self._check_for_meta_data(meta_data)
        edge_mask = self._check_edge_mask(edge_mask, node_id, condition_mask, meta_data)
        if x_o.shape[0] > 0:
            x_o = self._z_score_if_needed(x_o, node_id, condition_mask)
        
        sampling_kwargs = {**deepcopy(self.sampling_kwargs), **kwargs}
        if num_steps is None:
            num_steps = sampling_kwargs.pop("num_steps")
        else:
            if "num_steps" in sampling_kwargs:
                del sampling_kwargs["num_steps"]
 
        condition_mask = np.array(condition_mask) # Make sure it's will be treated as static by jit
        
        q = self._init_cns(x_o, num_steps, node_id=node_id, condition_mask=condition_mask, edge_mask=edge_mask, meta_data=meta_data, **kwargs)
        return q.log_prob(val)
    
    
    
    def sample_batched(self, num_samples, x_o, node_id=None, condition_mask=None, edge_mask=None, meta_data=None, num_steps=None, rng=None, **kwargs):
        @jax.vmap
        def get_batched_samples(keys,x_os):
            samples = self.sample(num_samples, x_os, node_id=node_id, condition_mask=condition_mask, edge_mask=edge_mask, meta_data=meta_data, num_steps=num_steps, rng=keys, **kwargs)
            return samples

        return get_batched_samples(jax.random.split(rng, x_o.shape[0]),x_o)

    def log_prob_batched(self, val, x_o, node_id=None, condition_mask=None, edge_mask=None, meta_data=None, num_steps=None, **kwargs):
        @jax.vmap
        def get_batched_log_probs(vals, x_os):
            log_probs = self.log_prob(vals, x_os, node_id=node_id, condition_mask=condition_mask, edge_mask=edge_mask, meta_data=meta_data, num_steps=num_steps, **kwargs)
            return log_probs

        return get_batched_log_probs(val,x_o)
    
    
    def compute_coverage_statistic(self, joint_samples, condition_mask=None, num_bins=20, rng=None, max_batch_sampling=None, max_batch_log_probs=250, sample_kwargs={}, log_prob_kwargs={"method":"euler"}):
        # Deprecated ...
        assert rng is not None, "Please provide a rng key"
        assert condition_mask is not None, "Please provide a condition mask"
        num_samples = joint_samples.shape[0]
        thetas = joint_samples[:,~condition_mask]
        xs = joint_samples[:,condition_mask]
        
        if max_batch_sampling is None:
            batched_samples = self.sample_batched(num_bins, x_o=xs, condition_mask=condition_mask, rng = rng, **sample_kwargs)
            print("Finished sampling")
        else:
            num_rounds = num_samples // max_batch_sampling
            batched_samples_per_round = []
            for i in range(num_rounds):
                xs_batch = xs[i*max_batch_sampling:(i+1)*max_batch_sampling]
                samples_batch = self.sample_batched(num_bins, x_o=xs_batch, condition_mask=condition_mask, rng = rng, **sample_kwargs)
                samples_batch = jax.device_put(samples_batch, jax.devices("cpu")[0])
                batched_samples_per_round.append(samples_batch)
                print("Finished sampling for batch ", i)
            batched_samples = jnp.concatenate(batched_samples_per_round, axis=0)
        
        batched_log_probs_true = self.log_prob_batched(thetas, x_o=xs, condition_mask=condition_mask, **log_prob_kwargs)
        batched_log_probs_true = jax.device_put(batched_log_probs_true, jax.devices("cpu")[0])
        print("Finished computing true log probs")
        
        num_rounds = num_samples // max_batch_log_probs
        log_prob_samples_per_round = []
        for i in range(num_rounds):
            xs_batch = xs[i*max_batch_log_probs:(i+1)*max_batch_log_probs]
            samples_batch = batched_samples[i*max_batch_log_probs:(i+1)*max_batch_log_probs]
            samples_batch = jax.device_put(samples_batch, jax.devices()[0])
            batched_log_probs = self.log_prob_batched(samples_batch, x_o=xs_batch, condition_mask=condition_mask, **log_prob_kwargs)
            batched_log_probs = jax.device_put(batched_log_probs, jax.devices("cpu")[0])
            log_prob_samples_per_round.append(batched_log_probs)
            print("Finished computing sample log probs for batch ", i)
        
        batched_samples = jax.device_put(batched_samples, jax.devices("cpu")[0])
        batched_log_probs_samples = jnp.concatenate(log_prob_samples_per_round, axis=0)
        
        alphas = np.linspace(1/num_bins, 1-1/num_bins, num_bins)
        covs = []
        for a in alphas:
            a = 1-a
            cov_a = jnp.mean(batched_log_probs_samples > jnp.percentile(batched_log_probs_true, a*100, axis=0))
            covs.append(cov_a)
        covs = jnp.array(covs)
        
        alphas = jnp.concatenate([jnp.array([0.]), alphas, jnp.array([1.])])
        covs = jnp.concatenate([jnp.array([0.]), covs, jnp.array([1.])])
        return alphas, covs
        
        

        
        
        


    def set_default_edge_mask_fn(self, edge_mask_fn):
        self.edge_mask_fn = edge_mask_fn

    def set_default_score_fn(self, score_fn):
        self.score_fn = score_fn
        
    def set_default_sampling_kwargs(self, **kwargs):
        self.sampling_kwargs = kwargs
        
    def set_default_meta_data(self, meta_data):
        self.meta_data = meta_data
        
    def _init_score(self, node_id, condition_mask, edge_mask, meta_data):

        def score_fn(t, x):
            score = self.score_fn(
                self.params,
                jnp.atleast_1d(t),
                x.reshape(-1, x.shape[-1], 1),
                node_id,
                condition_mask,
                meta_data=meta_data.reshape(-1, meta_data.shape[-1], 1) if meta_data is not None else None,
                edge_mask=edge_mask,
            ).reshape(x.shape)

            return score
        
        return score_fn

    def _init_backward_sde(self, node_id=None, condition_mask=None, edge_mask=None, meta_data=None):
        # Thats in general pretty shitty as it trickers recompilation every time we call sample
        def drift_backward(t, x):
            t = self.T_max - t

            score = self.score_fn(
                self.params,
                jnp.atleast_1d(t),
                x.reshape(-1, x.shape[-1], 1),
                node_id,
                condition_mask,
                meta_data=meta_data.reshape(-1, meta_data.shape[-1], 1) if meta_data is not None else None,
                edge_mask=edge_mask,
            ).reshape(x.shape)
            drift = self.sde.drift(t, x) - self.sde.diffusion(t, x) ** 2 * score
            return -drift.reshape(x.shape) * (1 - condition_mask.reshape(x.shape))

        def diffusion_backward(t, x):
            t = self.T_max - t
            return self.sde.diffusion(t, x).reshape(x.shape) * (
                1 - condition_mask.reshape(x.shape)
            )

        return drift_backward, diffusion_backward
    
    def _init_backward_ode(self, node_id=None, condition_mask=None, edge_mask=None, meta_data=None):
        def drift_backward(t, x):
            t = self.T_max - t
            score = self.score_fn(
                self.params,
                jnp.atleast_1d(t),
                x.reshape(-1, x.shape[-1], 1),
                node_id,
                condition_mask,
                edge_mask=edge_mask,
                meta_data=meta_data.reshape(-1, meta_data.shape[-1], 1) if meta_data is not None else None,
            ).reshape(x.shape)
            dx = self.sde.drift(t,x) - 0.5 * self.sde.diffusion(t, x) ** 2 * score
            return -dx.reshape(x.shape) * (1 - condition_mask.reshape(x.shape))

        return drift_backward
    
    def _init_cns(self, x_o, num_steps, node_id=None, condition_mask = None, edge_mask =None, meta_data=None, **kwargs):
        
        drift = self._init_backward_ode(node_id, condition_mask, edge_mask, meta_data)
        
        def drift_cond(t, x):
            xs = jnp.zeros((len(node_id),))
            xs = xs.at[condition_mask].set(x_o.reshape(-1))
            xs = xs.at[~condition_mask].set(x.reshape(-1))
            f = drift(t, xs)
            f = f[~condition_mask]
            return f 
        
        def f(x_T):
            y = odeint(drift_cond, x_T, jnp.linspace(0., self.T_max -self.T_min, num_steps),**kwargs)[-1]
            return y
        
        if node_id is None:
            q0 = Independent(Normal(self.marginal_end_mean[~condition_mask], self.marginal_end_std[~condition_mask]),1)
        else:
            q0 = Independent(Normal(self.marginal_end_mean[node_id][~condition_mask], self.marginal_end_std[node_id][~condition_mask]),1)
        q = TransformedDistribution(q0, f)
        return q
    
    
    def map(self, x_o, node_id=None, condition_mask=None, edge_mask=None, meta_data=None, num_init=1000, num_sampling_steps=None, init_learning_rate=1e-3, eps=0.1, rng=None, **kwargs):
        assert rng is not None, "Please provide a rng key"

        meta_data = self._check_for_meta_data(meta_data)
        edge_mask = self._check_edge_mask(edge_mask, node_id, condition_mask, meta_data)
        x_o = self._z_score_if_needed(x_o, node_id, condition_mask)
        
        if num_sampling_steps is None:
            num_sampling_steps = self.sampling_kwargs["num_steps"]
        
        score_fn = self._init_score(self.node_id, self.condition_mask, self.edge_mask, self.meta_data)
        
        latents_candidate = self.sample(num_init, x_o, node_id=node_id, condition_mask=condition_mask, meta_data=meta_data, edge_mask=edge_mask, rng=rng, num_steps=num_sampling_steps, **kwargs)
        
        x_o_repeated = jnp.repeat(x_o.reshape((-1,x_o.shape[-1])), num_init, axis=0)
        samples = jnp.concatenate([latents_candidate, x_o_repeated], axis=-1)
        
        optimizer = optax.adam(init_learning_rate)
        opt_state = optimizer.init(samples)

        def update(opt_state, xs):
            grads = -score_fn(jnp.ones((1,))*self.T_min, xs) * ~condition_mask
            updates, opt_state = optimizer.update(grads, opt_state)
            new_xs = optax.apply_updates(xs, updates)
            return new_xs, opt_state, grads
        
        def body_fn(state):
            xs, opt_state, grads = state 
            new_xs, opt_state, grads = update(opt_state, xs)
            return new_xs, opt_state, grads

        def cond_fn(state):
            _, _, grads = state
            return jnp.quantile(jnp.linalg.norm(grads, axis=-1), 0.2) > eps


        state = update(opt_state, samples)
        xs, opt_state, grads = jax.lax.while_loop(cond_fn, body_fn, state)
        
        latents = xs[:,~condition_mask]
        log_probs = self.log_prob(latents, x_o, node_id=node_id, condition_mask=condition_mask, meta_data=meta_data, edge_mask=edge_mask, num_steps=num_sampling_steps, **kwargs)
        
        idx = log_probs.argmax()
        best_latent = latents[idx]

        return best_latent,latents, log_probs
        
    
   
        

    def __getstate__(self) -> object:
        state = self.__dict__.copy()
        state["model_fn"] = None
        state["sde"] = None
        state["score_fn"] = None
        state["edge_mask_fn"] = None
        if self.z_score_params is not None:
            state["z_score_params"]["z_score_fn"] = None
            state["z_score_params"]["un_z_score_fn"] = None
        return state

    def __setstate__(self, state):
        from scoresbibm.methods.neural_nets import scalar_transformer_model
        from scoresbibm.methods.score_transformer import get_z_score_fn
        from scoresbibm.methods.sde import init_sde_related
        from scoresbibm.utils.edge_masks import get_edge_mask_fn
        from scoresbibm.tasks import get_task
        
        
        with jax.default_device(jax.devices("cpu")[0]):
            self.__dict__.update(state)
            self.sde, self.T_min, self.T_max, _, output_scale_fn = init_sde_related(
                **self.sde_init_params
            )
            _, self.model_fn = scalar_transformer_model(
                output_scale_fn=output_scale_fn, **self.model_init_params
            )
            self.score_fn = self.model_fn
            task_name = self.edge_mask_fn_params.get("task")
            task = get_task(task_name)
            self.edge_mask_fn = get_edge_mask_fn(
                self.edge_mask_fn_params["name"], task
            )
            
            if not hasattr(self, "z_score_params"):
                self.z_score_params = None
            
            if self.z_score_params is not None:
                z_score_fn, un_z_score_fn = get_z_score_fn(self.z_score_params["mean_per_node_id"], self.z_score_params["std_per_node_id"])
                self.z_score_params["z_score_fn"] = z_score_fn
                self.z_score_params["un_z_score_fn"] = un_z_score_fn
