import torch
import jax
import jax.numpy as jnp

import numpy as np
import math

import os
import pandas as pd

import pickle


def init_dir(dir_path: str):
    """Initializes a directory for storing models and summary.csv"""
    if not os.path.exists(dir_path + os.sep + "models"):
        os.makedirs(dir_path + os.sep + "models")

    if not os.path.exists(dir_path + os.sep + "summary.csv"):
        df = pd.DataFrame(
            columns=[
                "method",
                "task",
                "num_simulations",
                "seed",
                "model_id",
                "metric",
                "value",
                "time_train",
                "time_eval",
                "cfg",
            ]
        )
        df.to_csv(dir_path + os.sep + "summary.csv", index=False)


def query(
    name,
    task=None,
    method=None,
    num_simulations=None,
    metric=None,
    seed=None,
    model_id=None,
    value_statistic="mean",
    **kwargs,
):
    """Generic query function for querying the summary.csv file."""
    summary_df = get_summary_df(name)
    query = to_query_string("method", method)
    if num_simulations is not None:
        if query != "":
            query += "&"
        query += to_query_string("num_simulations", num_simulations)
    if seed is not None:
        if query != "":
            query += "&"
        query += to_query_string("seed", seed)
    if task is not None:
        if query != "":
            query += "&"
        query += to_query_string("task", task)
    if metric is not None:
        if query != "":
            query += "&"
        query += to_query_string("metric", metric)
    if model_id is not None:
        if query != "":
            query += "&"
        query += to_query_string("model_id", model_id)

    if query == "":
        df_q = summary_df
    else:
        df_q = summary_df.query(query)
    
    # Consistent with kwargs
    cfgs = df_q["cfg"].values
    cfgs = [eval(cfg) for cfg in cfgs]
    mask_include = []
    for cfg in cfgs:
        include = True
        for k, v in kwargs.items():
            include = include and check_query_cfg(cfg, k, v)
        mask_include.append(include)
        
    df_q = df_q[np.array(mask_include, dtype=bool)]
    

    # Evaluate value, which is a string
    df_q["value"] = df_q["value"].apply(lambda x: np.array(eval(x)) if isinstance(x, str) else np.array(x))
    if value_statistic == "mean":
        df_q["value"] = df_q["value"].apply(lambda x: np.mean(x) if x is not None else None)
    elif value_statistic == "median":
        df_q["value"] = df_q["value"].apply(lambda x: np.median(x) if x is not None else None)
    elif value_statistic == "std":
        df_q["value"] = df_q["value"].apply(lambda x: np.std(x) if x is not None else None)
    elif "quantile" in value_statistic:
        val = float(value_statistic.split("_")[1])
        df_q["value"] = df_q["value"].apply(lambda x: np.quantile(x, val))
    elif value_statistic == "none":
        pass
    else:
        raise NotImplementedError()
    return df_q

def check_query_cfg(cfg, query_str, query_value):
    levels = query_str.split("_")
    for level in levels:
        if level not in cfg:
            return True
        if isinstance(cfg, dict):
            cfg = cfg[level]
        else:
            return True
    
    return cfg == query_value




def to_query_string(name: str, var) -> str:
    """Translates a variable to string.

    Args:
        name (str): Query argument
        var (str): value

    Returns:
        str: Query == value ?
    """
    if var is None:
        return ""
    elif (
        var is pd.NA
        or var is torch.nan
        or var is math.nan
        or str(var) == "nan"
        or var is jnp.nan
    ):
        return f"{name}!={name}"
    elif isinstance(var, list) or isinstance(var, tuple):
        query = "("
        for v in var:
            if query != "(":
                query += "|"
            if isinstance(v, str):
                query += f"{name}=='{v}'"
            else:
                query += f"{name}=={v}"
        query += ")"
    else:
        if isinstance(var, str):
            query = f"{name}=='{var}'"
        else:
            query = f"{name}=={var}"
    return query


def get_summary_df(dir_path):
    """Returns the summary.csv file as a pandas dataframe."""
    df = pd.read_csv(dir_path + os.sep + "summary.csv")
    return df


def generate_unique_model_id(dir_path):
    """Generates a unique model id for saving a model."""
    summary_df = get_summary_df(dir_path)
    model_ids = summary_df["model_id"].values
    if len(model_ids) == 0:
        return 0
    elif len(model_ids) == 1:
        return 1
    else:
        max_id = np.max(model_ids)
        return max_id + 1


def save_model(model, dir_path, model_id):
    """Saves a model to a file."""
    file_name = dir_path + os.sep + "models" + os.sep + f"model_{model_id}.pkl"
    with open(file_name, "wb") as file:
        pickle.dump(model, file)


def save_summary(
    dir_path,
    method: str,
    task: str,
    num_simulations: int,
    model_id: int,
    metric: str,
    value: float,
    seed: int,
    time_train: float,
    time_eval: float,
    cfg: dict,
):
    """Saves a summary to the summary.csv file."""
    summary_df = get_summary_df(dir_path)
    new_row = pd.DataFrame(
        {
            "method": method,
            "task": task,
            "num_simulations": num_simulations,
            "seed": seed,
            "model_id": model_id,
            "metric": metric,
            "value": str(value),
            "time_train": str(time_train),
            "time_eval": str(time_eval),
            "cfg": str(cfg),
        },
        index=[len(summary_df)],
    )
    summary_df = pd.concat([summary_df, new_row], axis=0, ignore_index=True)
    summary_df.to_csv(dir_path + os.sep + "summary.csv", index=False)


def load_model(dir_path, model_id):
    """Loads a model from a file."""
    file_name = dir_path + os.sep + "models" + os.sep + f"model_{model_id}.pkl"
    with open(file_name, "rb") as file:
        return pickle.load(file)


def as_torch_tensor(*args):
    """Converts args to torch tensors."""
    updated_args = [torch.from_numpy(np.asarray(a)) for a in args]
    if len(updated_args) == 1:
        return updated_args[0]
    else:
        return updated_args


def as_jax_array(*args):
    updated_args = [jnp.asarray(np.asarray(a)) for a in args]
    if len(updated_args) == 1:
        return updated_args[0]
    else:
        return updated_args


def as_numpy_array(*args):
    """Converts args to numpy arrays."""
    updated_args = [np.asarray(a) for a in args]
    if len(updated_args) == 1:
        return updated_args[0]
    else:
        return updated_args
