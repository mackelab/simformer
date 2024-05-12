

from scoresbibm.utils.data_utils import query

import matplotlib as mpl
import matplotlib.pyplot as plt
from tueplots import bundles
import seaborn as sns

import os
import math
import numpy as np

_custom_styles = ["pyloric"]
_tueplot_styles = ["aistats2022", "icml2022", "jmlr2001", "neurips2021", "neurips2022"]
_mpl_styles = list(plt.style.available)

PATH = os.path.dirname(os.path.abspath(__file__))

DEFAULT_COLORS = {"npe": "#154c79", "nle": "#1e81b0", "nre": "#76b5c5", "nspe": "#fdae61", "score_transformer": "#d73027", "score_transformer_posterior": "#8b27d7", "score_transformer_directed": "#d74127","score_transformer_min_graphical": "#d74127", "score_transformer_undirected": "#911c1c"}

def get_style(style, **kwargs):
    if style in _mpl_styles:
        return [style]
    elif style in _tueplot_styles:
        return [getattr(bundles, style)(**kwargs)]
    elif style in _custom_styles:
        return [PATH + os.sep + style + ".mplstyle"]
    elif style == "science":
        return ["science"]
    elif style == "science_grid":
        return ["science", {"axes.grid": True}]
    elif style is None:
        return None
    elif style == "icml_science_grid":
        return  [getattr(bundles, "icml2022")(**kwargs), "science", {"axes.grid": True}]
    else:
        return style


class use_style:
    def __init__(self, style, kwargs={}) -> None:
        super().__init__()
        self.style = get_style(style) +  [kwargs]
        self.previous_style = {}

    def __enter__(self):
        self.previous_style = mpl.rcParams.copy()
        if self.style is not None:
            plt.style.use(self.style)

    def __exit__(self, *args, **kwargs):
        mpl.rcParams.update(self.previous_style)


def get_ylim_by_metric(metric):
    """ Get ylim by metric"""
    if "c2st" in metric:
        return (0.5, 1.0)
    else:
        return None
    
def get_metric_plot_name(metric):
    """ Get metric plot name"""
    if "c2st" in metric:
        return "C2ST"
    elif "nll" in metric:
        return "NLL"
    else:
        return metric
    
def get_task_plot_name(task):
    """ Get task plot name"""
    if task == "gaussian_linear":
        return "Linear Gaussian"
    elif task == "gaussian_mixture":
        return "Mixture Gaussian"
    elif task == "two_moons":
        return "Two Moons"
    elif task == "slcp":
        return "SLCP"
    elif task == "two_moons_all_cond":
        return "Two Moons (all cond.)"
    elif task == "slcp_all_cond":
        return "SLCP (all cond.)"
    elif task == "tree_all_cond":
        return "Tree (all cond.)"
    elif task == "marcov_chain_all_cond":
        return "HMM (all cond.)"
    elif task == "lotka_volterra":
        return "Lotka Volterra"
    elif task == "sir":
        return "SIR"
    else:
        return task

def get_method_plot_name(method):
    """ Get method plot name"""
    if method == "npe":
        return "NPE"
    elif method == "nle":
        return "NLE"
    elif method == "nre":
        return "NRE"
    elif method == "nspe":
        return "NSPE"
    elif method == "score_transformer":
        return "Simformer"
    elif method == "score_transformer_posterior":
        return "Simformer (posterior only)"
    elif method == "score_transformer_directed" or method == "score_transformer_min_graphical":  # Legacy support
        return "Simformer (directed graph)"
    elif method == "score_transformer_undirected" or method == "score_transformer_graphical":
        return "Simformer (undirected graph)"
    else:
        return method
    

def get_plot_name_fn(name):
    """ Get plot name fn"""
    if name == "method":
        return get_method_plot_name
    elif name == "task":
        return get_task_plot_name
    elif name == "metric":
        return get_metric_plot_name
    else:
        return lambda x:x


def use_all_plot_name_fn(name: str):
    return get_method_plot_name(get_task_plot_name(get_metric_plot_name(name)))

    
def float_to_power_of_ten(val: float):
    exp = math.log10(val)
    exp = int(exp)
    return rf"$10^{exp}$"


def plot_metric_by_num_simulations(name, method = None, task = None, num_simulations = None, seed = None, metric="c2st", with_error=False,  ax=None, figsize=(3, 2), color_map=None, hue=None,  df=None,**kwargs):
    """ Plot the metric"""
    
    if df is None:
        if with_error:
            df = query(name, task=task, method=method, num_simulations=num_simulations, metric=metric, seed=seed, value_statistic="none",  **kwargs)
            values = df["value"].apply(lambda x: np.quantile(x, 0.5))
            error_upper = df["value"].apply(lambda x: np.quantile(x, 0.95))
            error_lower = df["value"].apply(lambda x: np.quantile(x, 0.05))
        else:
            df = query(name, task=task, method=method, num_simulations=num_simulations, metric=metric, seed=seed, **kwargs)
    else:
        df = df.copy()
    
    
    ylims = get_ylim_by_metric(metric)
    df = df.sort_values(by=["num_simulations"])
    
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
    else:
        fig = None
    
    if with_error:
        pass
    else:
        sns.pointplot(x="num_simulations", y="value", data=df, ax=ax, marker=".", dodge=False, hue=hue, palette=color_map, alpha=kwargs.get("alpha", 0.5))
    ax.set_xlabel("Number of simulations")

    ax.set_xticks(range(len(df["num_simulations"].unique())))
    ax.set_xticklabels(
            [float_to_power_of_ten(float(a._text)) for a in ax.get_xticklabels()]
        )
    if ylims is not None:
        ax.set_ylim(*ylims)
    ax.set_ylabel(get_metric_plot_name(metric))
    
    return fig, ax




def plot_expected_coverage(name, method = None, task = None, num_simulations = None, seed = None, metric="c2st", ax=None, figsize=(2, 2), color_map=None, alpha=0.5,  df=None,**kwargs):
    if df is None:
        df = query(name, task=task, method=method, num_simulations=num_simulations, metric=metric, seed=seed, value_statistic="none",  **kwargs)
    else:
        df = df.copy()
    
    num_simulations = df["num_simulations"].values
    values = df["value"].values
    names = df["metric"].values
    alphas = values
    covs = values
    
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
    else:
        fig = None
    
    for n, v in zip(names, values):
        if "cov" in n:
            alphas = v[0]
            covs = v[1]
            if color_map is not None:
                color = color_map[n]
            
            
            ax.plot(alphas, covs, lw=2, alpha=alpha)
    ax.plot([0, 1], [0, 1], "--", color="black", lw=1, alpha=0.5) 

    
    num_simulation = num_simulations[0]
    if fig is not None:
        fig.suptitle(get_method_plot_name(method) + f" [{float_to_power_of_ten(num_simulation)} sims.]", y=1.05)
    
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.set_xticks([0., 0.5, 1.])
    ax.set_yticks([0., 0.5, 1.])
    
    ax.set_xlabel("Credibility level")
    ax.set_ylabel("Empirical coverage")
    
    return fig, ax
    


def get_sorting_key_fn(name):
    if name == "method":
        def key_fn(method):
            if method == "npe":
                return 0
            elif method == "nle":
                return 1
            elif method == "nre":
                return 2
            elif  "score_transformer" in method:
                return 3
            else:
                return 4
        return np.vectorize(key_fn)
    elif name == "task":
        def key_fn(task):
            if task == "gaussian_linear" or "tree" in task:
                return 0
            elif task == "gaussian_mixture" or "marcov" in task:
                return 1
            elif task == "two_moons" or task == "two_moons_all_cond":
                return 2
            elif task == "slcp" or task == "two_moons_all_cond":
                return 3
            else:
                return 4
        return np.vectorize(key_fn)
    else:
        return lambda x:x 
    
    
def multi_plot(
    name,
    cols,
    rows,
    plot_fn,
    fig_title=None,
    y_label_by_row=True,
    y_labels=None,
    scilimit=3,
    x_labels=None,
    y_lims=None,
    fontsize_title=None,
    figsize_per_row=2,
    figsize_per_col=2.3,
    legend_bbox_to_anchor=[0.5, -0.1],
    legend_title=False,
    legend_ncol=10,
    legend_kwargs={},
    fig_legend=True,
    df = None,
    verbose=False,
    **kwargs,
):
    if df is None:
        df = query(name, **kwargs)
    else:
        df = df.copy()

    df = df.sort_values(cols, na_position="first", key=get_sorting_key_fn(cols))
    cols_vals = df[cols].dropna().unique()

    df = df.sort_values(rows, na_position="first", key=get_sorting_key_fn(rows))
    rows_vals = df[rows].dropna().unique()

    # Creating a color map if hue is specified:
    if "hue" in kwargs and "color_map" not in kwargs:
        hue_col = kwargs["hue"]
        df = df.sort_values(
            hue_col, na_position="first", key=get_sorting_key_fn(hue_col)
        )
        unique_vals = df[hue_col].unique()
        unique_vals.sort()
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        color_map = {}
        for i in range(len(unique_vals)):
            color_map[unique_vals[i]] = colors[min(i, len(colors) - 1)]
    else:
        if "color_map" not in kwargs:
            color_map = None
        else:
            color_map = kwargs.pop("color_map")


    n_cols = len(cols_vals)
    n_rows = len(rows_vals)

    if n_cols == 0:
        raise ValueError(f"No columns found in the dataset with label {cols}")

    if n_rows == 0:
        raise ValueError(f"No rows found in the dataset with label {rows}")

    figsize = (n_cols * figsize_per_col, n_rows * figsize_per_row)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=True)

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    else:
        if n_cols == 1:
            axes = np.array([[ax] for ax in axes])

        if n_rows == 1:
            axes = np.array([axes])

    max_legend_elements = 0

    for i in range(n_rows):
        for j in range(n_cols):

            axes[i, j].ticklabel_format(axis="y", scilimits=[-scilimit, scilimit])
            if y_labels is not None:
                y_label = y_labels[i]
            else:
                if y_label_by_row:
                    name_fn = get_plot_name_fn(rows)
                    y_label = name_fn(rows_vals[i])
                else:
                    y_label = None

            if x_labels is not None:
                x_label = x_labels[i]
            else:
                x_label = None

            if y_lims is not None:
                if isinstance(y_lims, tuple):
                    y_lim = y_lims
                else:
                    if isinstance(y_lims[0], tuple):
                        y_lim = y_lims[i]
                    else:
                        if isinstance(y_lims[0, 0], tuple):
                            y_lim = y_lims[i, j]
                        else:
                            raise ValueError()
            else:
                y_lim = None

            plot_dict = {cols: cols_vals[j], rows: rows_vals[i]}
            plot_kwargs = {**kwargs, **plot_dict}

            if verbose:
                print(plot_kwargs)
            try:
                plot_fn(name, ax=axes[i, j], color_map=color_map, **plot_kwargs)
            except Exception as e:
                if verbose:
                    print(str(e))
                    # Print traceback
                    import traceback
                    traceback.print_exc()
                    

            if y_label is not None:
                axes[i, j].set_ylabel(y_label)
                axes[i, j].yaxis.set_label_coords(-0.3, 0.5)
            else:
                fn = get_plot_name_fn(cols)
                y_label = axes[i, j].get_ylabel()
                axes[i, j].set_ylabel(fn(y_label))
                axes[i, j].yaxis.set_label_coords(-0.3, 0.5)

            if x_label is not None:
                axes[i, j].set_xlabel(x_label)
            else:
                fn = get_plot_name_fn(rows)
                x_label = axes[i, j].get_xlabel()
                axes[i, j].set_xlabel(fn(x_label))
            if i == 0:
                name_fn = get_plot_name_fn(cols)
                axes[i, j].set_title(name_fn(cols_vals[j]))

            if i < n_rows - 1:
                axes[i, j].set_xlabel(None)
                axes[i, j].set_xticklabels([])

            if j > 0:
                axes[i, j].set_ylabel(None)

            if y_lim is not None:
                axes[i, j].set_ylim(y_lim)

            if i > 0:
                axes[i, j].set_title(None)

            if axes[i, j].get_legend() is not None:
                legend = axes[i, j].get_legend()
                if len(legend.get_texts()) > max_legend_elements:
                    max_legend_elements = len(legend.get_texts())
                    legend_text = [t._text for t in legend.get_texts()]
                    if legend_title:
                        legend_title = legend.get_title()._text
                    else:
                        legend_title = ""
                    legend_handles = legend.legendHandles
                legend.remove()

    for i in range(n_rows):
        for j in range(n_cols):
            if len(axes[i, j].lines) == 0 and len(axes[i, j].collections) == 0:
                axes[i, j].text(
                    0.5,
                    0.5,
                    "No data",
                    bbox={
                        "facecolor": "white",
                        "alpha": 1,
                        "edgecolor": "none",
                        "pad": 1,
                    },
                    ha="center",
                    va="center",
                )

    if fig_legend and "legend_text" in locals() and len(legend_text) > 0:
        text = [use_all_plot_name_fn(t) for t in list(dict.fromkeys(legend_text))]
        handles = list(dict.fromkeys(legend_handles))
        fig.legend(
            labels=text,
            handles=handles,
            title=use_all_plot_name_fn(str(legend_title)),
            ncol=legend_ncol,
            loc="lower center",
            bbox_to_anchor=legend_bbox_to_anchor,
            **legend_kwargs,
        )

    fig.tight_layout()
    if fig_title is not None:
        fig.suptitle(fig_title)
    return fig, axes