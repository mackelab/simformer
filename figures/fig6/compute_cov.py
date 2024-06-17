# %%
from scoresbibm.utils.data_utils import query, get_summary_df,load_model
from scoresbibm.utils.plot import plot_metric_by_num_simulations, use_style,multi_plot

from scoresbibm.tasks.unstructured_tasks import SIRTask, MultivariateNormal
from probjax.distributions.transformed_distribution import TransformedDistribution
from probjax.core import rv

import matplotlib.pyplot as plt
import seaborn as sns

import jax
import jax.numpy as jnp
import os

import logging
logging.getLogger('matplotlib.font_manager').disabled = True

def main():
    # %%
    task = SIRTask(time_end=40)

    observation_generator = task.get_observation_generator()
    reference_sampler = task.get_reference_sampler()
    observation_stream = observation_generator(jax.random.PRNGKey(8))

    # %%
    COLOR_INFECTED = "darkorange"
    COLOR_RECOVERED = "seagreen"
    COLOR_DEATHS = "C3"
    COLOR_PARAMETER = "navy"

    # %%
    PATH = "../../results/bm_sir"

    # %%
    df = query(PATH, method_sde_name="vesde", method="score_transformer", num_simulations=100000)

    # %%
    model = load_model(PATH, df["model_id"].iloc[0])


    # %%
    import numpy as np  


    # %%
    joint_stream = task.get_observation_generator("posterior")(jax.random.PRNGKey(42))

    # %%
    log_probs_q = []
    log_probs_true_theta = []
    for i in range(200):
        print("Finished", i)
        condition_mask, x_o, theta_o, meta_data, node_id = next(joint_stream)
        samples = model.sample(500, x_o=x_o, condition_mask=condition_mask, meta_data=meta_data,node_id=node_id, rng=jax.random.PRNGKey(i))
        log_probs = model.log_prob(samples, x_o=x_o, condition_mask=np.array(condition_mask), node_id=node_id, meta_data=meta_data, num_steps=100)
        log_probs_true = model.log_prob(theta_o, x_o=x_o, condition_mask=np.array(condition_mask), node_id=node_id, meta_data=meta_data, num_steps=100)
        log_probs_q.append(log_probs)
        log_probs_true_theta.append(log_probs_true)

    # %%
    log_probs_q_vec = jnp.stack(log_probs_q)
    log_probs_true_vec = jnp.stack(log_probs_true_theta)
    
    jnp.save("log_probs_q.npy", log_probs_q_vec)
    jnp.save("log_probs_true.npy", log_probs_true_vec)

  

if __name__ == "__main__":
    main()
