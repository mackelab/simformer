

from scoresbibm.tasks.base_task import InferenceTask, AllConditionalTask

import jax
import time
import numpy as np
import torch

from scoresbibm.utils.condition_masks import get_condition_mask_fn



def eval_inference_task(task: InferenceTask, model, metric_fn, metric_params, rng, **kwargs):
    metric_values = []
    metric_params = dict(metric_params)
    condition_mask_fn = metric_params.pop("condition_mask_fn", "structured_random")
    if condition_mask_fn != "posterior":
        return None, None
    average_sampling_time = 0
    for i in task.observations:
        rng_metric, rng_metric_i = jax.random.split(rng)
        x_o = task.get_observation(i)
        true_posterior_samples = task.get_reference_posterior_samples(i)
        start_time = time.time()
        est_posterior_samples = model.sample(num_samples=true_posterior_samples.shape[0], x_o=x_o, rng=rng_metric_i)
        sampling_time = time.time() - start_time
        val = metric_fn(true_posterior_samples, est_posterior_samples, rng=rng_metric, **metric_params)
        print("Metric value: ", val)
        metric_values.append(val)
        average_sampling_time += sampling_time / len(task.observations)
    return metric_values, average_sampling_time


def eval_all_conditional_task(task: AllConditionalTask, model, metric_fn, metric_params, rng, num_samples=2000, num_evaluations=2):
    metric_values = []
    average_sampling_time = 0
    metric_params = dict(metric_params)
    condition_mask_fn = metric_params.pop("condition_mask_fn", "structured_random")
    reference_sampler = task.get_reference_sampler()
    observation_generator = task.get_observation_generator(condition_mask_fn=condition_mask_fn)
    
    rng, rng_obs = jax.random.split(rng)
    observation_stream = observation_generator(rng_obs)
    for i in range(num_evaluations):
        rng, rng_metric, rng_sample_ref, rng_sample_model = jax.random.split(rng,4)
        condition_mask, x_o, theta_o = next(observation_stream)
        print("Conditional: ", condition_mask,x_o)
        start_time = time.time()
        est_posterior_samples = model.sample(num_samples, x_o= x_o,condition_mask=condition_mask, rng=rng_sample_model)
        sampling_time = time.time() - start_time
        true_posterior_samples = reference_sampler.sample(num_samples, x_o= x_o,condition_mask=condition_mask,rng=rng_sample_ref)  
        metric_value = metric_fn(est_posterior_samples, true_posterior_samples, rng=rng_metric, **metric_params)
        print("Metric value: ", metric_value)
        metric_values.append(metric_value)
        average_sampling_time += sampling_time / num_evaluations
    return metric_values, average_sampling_time

def eval_unstructured_task(task, model, metric_fn, metric_params, rng, num_samples=1000, num_evaluations=50):
    metric_values = []
    average_sampling_time = 0
    metric_params = dict(metric_params)
    condition_mask_fn = metric_params.pop("condition_mask_fn", "structured_random")
    reference_sampler = task.get_reference_sampler()
    observation_generator = task.get_observation_generator(condition_mask_fn=condition_mask_fn)
    
    rng, rng_obs = jax.random.split(rng)
    observation_stream = observation_generator(rng_obs)
    for i in range(num_evaluations):
        rng, rng_metric, rng_sample_ref, rng_sample_model = jax.random.split(rng,4)
        condition_mask, x_o, theta_o, meta_data, node_id  = next(observation_stream)
        print("Conditional: ", condition_mask,x_o)
        start_time = time.time()
        est_posterior_samples = model.sample(num_samples, x_o= x_o,condition_mask=condition_mask,node_id=node_id, meta_data=meta_data, rng=rng_sample_model)
        sampling_time = time.time() - start_time
        true_posterior_samples = reference_sampler.sample(num_samples, x_o= x_o,condition_mask=condition_mask,meta_data=meta_data,rng=rng_sample_ref)  
        metric_value = metric_fn(est_posterior_samples, true_posterior_samples, rng=rng_metric, **metric_params)
        print("Metric value: ", metric_value)
        metric_values.append(metric_value)
        average_sampling_time += sampling_time / num_evaluations
    return metric_values, average_sampling_time


def eval_negative_log_likelihood(task, model, metric_params, rng):
    condition_mask_fn = get_condition_mask_fn(metric_params["condition_mask_fn"])
    condition_mask = condition_mask_fn(rng, 1, task.get_theta_dim(), task.get_x_dim())[0]
    

    num_samples = metric_params["num_samples"]
    num_steps = metric_params["num_steps"]
    
    data = task.get_data(num_samples, rng=rng)
    thetas, xs = data["theta"], data["x"]
    batch_size = metric_params["batch_size"]
    thetas = np.array(thetas)
    xs = np.array(xs)
    

    if model.backend == "jax":
        data = jax.numpy.concatenate([thetas, xs], axis=1)
        
        latents = data[:, ~condition_mask]
        observed = data[:, condition_mask]
        @jax.jit
        def ll_batch(latents, observed):
            return  model.log_prob_batched(latents, observed, condition_mask=condition_mask, num_steps=num_steps)
    elif model.backend == "torch":
        data = torch.cat([torch.tensor(thetas), torch.tensor(xs)], dim=1)
        condition_mask = torch.tensor(np.array(condition_mask))
        latents = data[:, ~condition_mask]
        observed = data[:, condition_mask]
        def ll_batch(latents, observed):
            return model.log_prob_batched(latents, observed, condition_mask=condition_mask, num_steps=num_steps).detach().numpy()

    start_time = time.time()
    num_batches = num_samples // batch_size
    nlls = []
    for i in range(num_batches):
        latents_batch = latents[i*batch_size:(i+1)*batch_size]
        observed_batch = observed[i*batch_size:(i+1)*batch_size]
        log_likelihoods = ll_batch(latents_batch, observed_batch)
        nlls.append(-log_likelihoods)
    nlls = np.concatenate(nlls)
    nlls = [float(nll) for nll in nlls]
    eval_time = time.time() - start_time
    return nlls, eval_time
    
    
def eval_coverage(task, model, metric_params,rng):
    
    condition_mask_fn = get_condition_mask_fn(metric_params["condition_mask_fn"])
    condition_mask = condition_mask_fn(rng, 1, task.get_theta_dim(), task.get_x_dim())[0]
    

    num_samples = metric_params["num_samples"]
    num_bins = metric_params["num_bins"]
    sample_kwargs = metric_params["sample_kwargs"]
    log_prob_kwargs = metric_params["log_prob_kwargs"]
    
    data = task.get_data(num_samples, rng=rng)
    thetas, xs = data["theta"], data["x"]
    batch_size = metric_params["batch_size"]
    thetas = np.array(thetas)
    xs = np.array(xs)
    
    if model.backend == "jax":
        data = jax.numpy.concatenate([thetas, xs], axis=1) 
        thetas = data[:, ~condition_mask]
        xs = data[:, condition_mask]
        condition_mask = np.array(condition_mask)
  
        def sample_fn(rng,x_o):
            samples = model.sample_batched(num_bins, x_o=x_o, condition_mask=condition_mask, rng = rng, **sample_kwargs)
            return samples
        
        @jax.jit
        def log_prob_fn(samples, x_o):
            return model.log_prob_batched(samples, x_o=x_o, condition_mask=condition_mask, **log_prob_kwargs)

    elif model.backend == "torch":
        data = torch.cat([torch.tensor(thetas), torch.tensor(xs)], dim=1)
        condition_mask = torch.tensor(np.array(condition_mask))
        thetas = data[:, ~condition_mask]
        xs = data[:, condition_mask]
        
        def sample_fn(rng,x_o):
            samples = model.sample_batched(num_bins, x_o=x_o, condition_mask=condition_mask, rng = rng, **sample_kwargs)
            return samples.detach().numpy()
        
        def log_prob_fn(samples, x_o):
            return model.log_prob_batched(samples, x_o=x_o, condition_mask=condition_mask, **log_prob_kwargs).detach().numpy()
        
        

    num_rounds = num_samples // batch_size
    batched_samples_per_round = []
    for i in range(num_rounds):
        rng, rng_sample = jax.random.split(rng)
        xs_batch = xs[i*batch_size:(i+1)*batch_size]
        samples_batch = sample_fn(rng_sample, xs_batch)
        samples_batch = np.array(samples_batch)
        batched_samples_per_round.append(samples_batch)
        print("Finished sampling for batch ", i)
    batched_samples = np.concatenate(batched_samples_per_round, axis=0)
    
    batched_log_probs_true = log_prob_fn(thetas, xs)
    batched_log_probs_true = np.array(batched_log_probs_true)
    print("Finished computing true log probs")
    
    num_rounds = num_samples // batch_size
    log_prob_samples_per_round = []
    for i in range(num_rounds):
        xs_batch = xs[i*batch_size:(i+1)*batch_size]
        samples_batch = batched_samples[i*batch_size:(i+1)*batch_size]
        batched_log_probs = log_prob_fn(samples_batch, xs_batch)
        batched_log_probs = np.array(batched_log_probs)
        log_prob_samples_per_round.append(batched_log_probs)
        print("Finished computing sample log probs for batch ", i)

    batched_log_probs_samples = np.concatenate(log_prob_samples_per_round, axis=0)
    
    alphas = np.linspace(1/num_bins, 1-1/num_bins, num_bins)
    covs = []
    for a in alphas:
        a = 1-a
        cov_a = np.mean(batched_log_probs_samples > np.percentile(batched_log_probs_true, a*100, axis=0))
        covs.append(cov_a)
    covs = np.array(covs)
    
    alphas = np.concatenate([np.array([0.]), alphas, np.array([1.])])
    covs = np.concatenate([np.array([0.]), covs, np.array([1.])])
    alphas = [float(a) for a in alphas]
    covs = [float(c) for c in covs]
    return (alphas, covs), None