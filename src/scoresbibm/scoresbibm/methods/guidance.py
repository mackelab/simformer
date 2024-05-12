import jax 
import jax.numpy as jnp

from probjax.utils.sdeint import register_method
from functools import partial


def repaint(model,key, condition_mask,x_o, x_T,num_steps=500, resampling_steps=10, node_id=None, edge_mask=None, meta_data=None):
    
    ts = jnp.linspace(model.T_min, model.T_max, num_steps)
    score_fn = model._init_score(node_id = node_id,condition_mask=jnp.zeros_like(condition_mask), edge_mask=edge_mask, meta_data=meta_data)
    x_o_pad = jnp.zeros_like(x_T)
    x_o_pad = x_o_pad.at[condition_mask].set(x_o)
    x_o = x_o_pad
    
    def scan_fn(carry, t0):
        key,t1, x1 = carry 
        dt = t0 - t1
        #print(dt)
        key, subkey = jax.random.split(key)
        # Sample from the marginal distirbution at t0, given known values x_o
        x0_mean = model.sde.marginal_mean(t0, x_o)
        x0_std = model.sde.marginal_stddev(t0, x_o)
        x0_known = x0_mean + x0_std * jax.random.normal(subkey, shape=x1.shape)
        # Predict the unknown values at t0 given values from t1
        key, subkey = jax.random.split(key)
        score = score_fn(t1, x1)
        drift_backward = model.sde.drift(t1, x1) -  model.sde.diffusion(t1, x1)**2 * score
        diffusion_backward = model.sde.diffusion(t1, x1) 
        x0_unknown = x1 + drift_backward * dt + diffusion_backward * jnp.sqrt(jnp.abs(dt)) * jax.random.normal(subkey, shape=x1.shape)
        x0 = condition_mask * x0_known + (1 - condition_mask) * x0_unknown
        
        for _ in range(resampling_steps):
            # Forward sample from the SDE
            key, subkey = jax.random.split(key)
            x1 = x0 - model.sde.drift(t0, x0) * dt + model.sde.diffusion(t0, x0) * jnp.sqrt(jnp.abs(dt)) * jax.random.normal(subkey, shape=x1.shape)
            
            # Update again
            key, subkey = jax.random.split(key)
            # Sample from the marginal distirbution at t0, given known values x_o
            x0_mean = model.sde.marginal_mean(t0, x_o)
            x0_std = model.sde.marginal_stddev(t0, x_o)
            x0_known = x0_mean + x0_std * jax.random.normal(subkey, shape=x1.shape)
            # Predict the unknown values at t0 given values from t1
            key, subkey = jax.random.split(key)
            score = score_fn(t1, x1)
            drift_backward = model.sde.drift(t1, x1) - model.sde.diffusion(t1, x1)**2 * score
            diffusion_backward = model.sde.diffusion(t1, x1) 
            x0_unknown = x1 + drift_backward * dt + diffusion_backward * jnp.sqrt(jnp.abs(dt)) * jax.random.normal(subkey, shape=x1.shape)
            x0 = condition_mask * x0_known + (1 - condition_mask) * x0_unknown
            
        return (key, t0, x0), x0
    
    
    carry = (key, ts[-1], x_T)
    _, x_T = jax.lax.scan(scan_fn, carry, ts[::-1][1:])
    return x_T[-1]

def generalized_guidance(model,constraint_fn,key, condition_mask, x_T, num_steps=500, resampling_steps=0, node_id=None, edge_mask=None, meta_data=None):
    
    ts = jnp.linspace(model.T_min, model.T_max, num_steps)
    score_fn = model._init_score(node_id = node_id,condition_mask=condition_mask, edge_mask=edge_mask, meta_data=meta_data)
    constraint_score_fn = jax.grad(constraint_fn, argnums=0)
    
    def scan_fn(carry, t0):
        key,t1, x1 = carry 
        dt = t0 - t1
        # Predict the unknown values at t0 given values from t1
        key, subkey = jax.random.split(key)
        score = score_fn(t1, x1)
        x_tweedy_estimator = (x1 + model.sde.marginal_stddev(t1, jnp.array([1.]))**2 * score)/model.sde.marginal_mean(t1, jnp.array([1.])) # Predict x0
        constraint_score = constraint_score_fn(x_tweedy_estimator, t1)
        score = score + constraint_score
        drift_backward = (1-condition_mask)*(model.sde.drift(t1, x1) - model.sde.diffusion(t1, x1)**2 * score)
        diffusion_backward = (1-condition_mask)*(model.sde.diffusion(t1, x1) )
        x0 = x1 + drift_backward * dt + diffusion_backward * jnp.sqrt(jnp.abs(dt)) * jax.random.normal(subkey, shape=x1.shape)
        
        for _ in range(resampling_steps):
            # Forward sample from the SDE
            key, subkey = jax.random.split(key)
            x1 = x0 -  (1-condition_mask)*(model.sde.drift(t0, x0) * dt - model.sde.diffusion(t0, x0) * jnp.sqrt(jnp.abs(dt)) * jax.random.normal(subkey, shape=x1.shape))
            
            # Update again
            key, subkey = jax.random.split(key)
            score = score_fn(t1, x1)
            x_tweedy_estimator = (x1 + model.sde.marginal_stddev(t1, jnp.array([1.]))**2 * score)/model.sde.marginal_mean(t1, jnp.array([1.])) # Predict x0
            constraint_score = constraint_score_fn(x_tweedy_estimator,t1)
            score = score + constraint_score
            drift_backward = (1-condition_mask)*(model.sde.drift(t1, x1) -  model.sde.diffusion(t1, x1)**2 * score)
            diffusion_backward = (1-condition_mask)*model.sde.diffusion(t1, x1) 
            x0 = x1 + drift_backward * dt + diffusion_backward * jnp.sqrt(jnp.abs(dt)) * jax.random.normal(subkey, shape=x1.shape)
            
        return (key, t0, x0), x0
    
    
    carry = (key, ts[-1], x_T)
    _, x_T = jax.lax.scan(scan_fn, carry, ts[::-1][1:])
    return x_T[-1]

    
    
def register_classifier_free_guidance(model, old_condition_mask, x_o, likelihood_scale=3., prior_scale=1.):
    
    def classifier_free_score_fn(params ,t ,x ,node_id ,condition_mask ,meta_data=None,edge_mask=None):
        # Zero out condition_mask
        unconditional_score = model.model_fn(params,t,x,node_id,jnp.zeros_like(condition_mask),meta_data=meta_data,edge_mask=edge_mask)
        x_conditional = x.at[...,old_condition_mask,0].set(x_o.reshape(-1))
        conditional_score = model.model_fn(params,t,x_conditional,node_id,old_condition_mask,meta_data=meta_data,edge_mask=edge_mask)
        condition_mask = old_condition_mask.reshape(unconditional_score.shape)
        likelihood_part = (conditional_score - unconditional_score)*(1 - condition_mask)
        prior_part = unconditional_score
        
        return likelihood_scale * likelihood_part + prior_scale * prior_part
    
    model.score_fn = classifier_free_score_fn
        
        

def register_naive_inpaint_guidance(model, condition_mask, x_o):
    def unconditional_score_fn(params ,t ,x ,node_id ,condition_mask ,meta_data=None,edge_mask=None):
        # Zero out condition_mask
        condition_mask = jnp.zeros_like(condition_mask)
        return model.model_fn(params,t,x,node_id,condition_mask,meta_data=meta_data,edge_mask=edge_mask)
    model.score_fn = unconditional_score_fn
    

def register_generalized_guidance(model, condition_mask, x_o, score_manipulator="conditional", **score_manipulator_kwargs):
    score_manipulator = get_constraint_fn(score_manipulator, **score_manipulator_kwargs)
    def additive_score_fn(params ,t ,x ,node_id ,local_condition_mask ,meta_data=None,edge_mask=None):
        # Zero out condition_mask
        joint_score = model.model_fn(params,t,x,node_id,local_condition_mask | condition_mask,meta_data=meta_data,edge_mask=edge_mask)
        manipulation_score = score_manipulator(t,x,condition_mask,x_o)
        manipulation_score = manipulation_score.reshape(joint_score.shape)
        #print(joint_score, manipulation_score)  
        return joint_score + manipulation_score
    
    model.score_fn = additive_score_fn
    
def get_constraint_fn(name, **kwargs):
    if name == "interval":
        return lambda x,t: log_step_fn(x,t,**kwargs)
    elif name == "linear":
        return lambda x,t: log_linear_fn_approximation(x,t,**kwargs)  
    elif name == "conditional":
        return lambda x,t: log_conditional_fn(x,t,**kwargs)
    elif name == "polytope":
        return lambda x,t: log_polytope_fn_approximation(x,t,**kwargs)
    else:
        raise NotImplementedError(f"Score manipulator {name} not implemented")



# Numerical stability! (Sigmoid is numerically unstable for large values) -> Use log_sigmoid
def log_step_fn(x,t,constraint_mask, x_o,a=None,b=None, scaling_fn=lambda x: 1/x):
    scale = scaling_fn(t)
    x = x.reshape(x.shape[0],-1)
    constraint_mask = constraint_mask.reshape(x.shape)
    if a is None:
        x1 = 0.
    else:
        x1 = jax.nn.log_sigmoid(scale * (x - a))*constraint_mask
    if b is None:
        x2 = 0.
    else:
        x2 = jax.nn.log_sigmoid(scale * (b- x))*constraint_mask
    return jnp.sum(x1 + x2)

def log_linear_fn_approximation(x,t,constraint_mask, x_o, a , scaling_fn):
    scale = scaling_fn(t)
    x = x.reshape(-1)
    x1 = jax.nn.log_sigmoid(scale * (jnp.sum(x * a)))
    x2 = jax.nn.log_sigmoid(-scale * (jnp.sum(x * a)))
    return jnp.sum(x1 + x2)

def log_conditional_fn(x,t, constraint_mask, x_o, scaling_fn):
    scale = scaling_fn(t)
    x_cond = x.at[constraint_mask].set(x_o)
    x1 = -jnp.sum(scale*(constraint_mask*(x - x_cond)**2))
    return jnp.sum(x1)
    
def smaller_equal_constraint_fn(x,t,constraint_mask, x_o, constrating_fn, constrain_value, scaling_fn):
    scale = scaling_fn(t)
    constraint = jax.nn.relu(scale * (constrating_fn(x) - constrain_value)).max(axis=-1)
    constraint = jax.nn.log_sigmoid(-constraint) 
    return jnp.sum(constraint)

def log_polytope_fn_approximation(x,t,condition_mask, x_o ,A , scaling_fn):
    return jnp.sum(smaller_equal_constraint_fn(t,x,condition_mask, x_o, lambda x: x@A.T, 1., scaling_fn=scaling_fn))
