

try:
    from sbi.analysis import pairplot
except:
    pass

import matplotlib as mpl
import matplotlib.pyplot as plt


import numpy as np
import jax
import jax.numpy as jnp


import os
import pkg_resources
import time

def is_installed(package_name):
    try:
        pkg_resources.get_distribution(package_name)
        return True
    except pkg_resources.DistributionNotFound:
        return False

def install_packages():
    if not is_installed("probjax"):
        print("Incomplete installation. Installing packages...")

        os.system("pip install jax[cuda12]==0.4.23")
        os.system("pip install nvidia-cudnn-cu12==8.9.7.29") # Fix for jaxlib
        os.system("pip install -e simformer/src/probjax")
        os.system("pip install -e simformer/src/scoresbibm")
        # First attempt can fail in colab for some reason
        os.system("pip install -e simformer/src/scoresbibm")
        os.system("pip install ipympl -q --root-user-action=ignore")

        print("Installation complete")
        print("Killing Kernel for restart")
        time.sleep(0.5)
        os.kill(os.getpid(), 9)
    else:
        os.system("pip install ipympl -q --root-user-action=ignore")



lowers = np.array([-1.009, -1.005, -1.153, -1.452])
uppers = np.array([1.009,  1.017, 0.367, 1.478])
# Define a function to update the plot for each frame


def interactive_pairplot(sample_fn, init_condition_mask, lowers, uppers, labels=None,figsize=(8,8), off_diag_bins=60, diag_bins=20):
    global j
    global condition_mask
    global x

    condition_mask = init_condition_mask
    x = jnp.zeros_like(condition_mask, dtype=jnp.float32)
    j = 0

    def update_axes(samples, axes):

        global x

        for i in range(len(axes)):
            for j in range(i+1,len(axes)):
                if i != j:
                    axes[i,j].clear()
                    axes[i,j].hist2d(samples[:,j], samples[:,i], bins=off_diag_bins,range=((lowers[j], uppers[j]),(lowers[i], uppers[i])))


            for child in axes[i,i].get_children():
                if isinstance(child, mpl.lines.Line2D):
                    child.remove()
            # Get histogram data - this for some reason is orders of magnitude faster than plt.hist...
            hist, bin_edges = np.histogram(samples[:,i], bins="auto", density=True, range=(lowers[i], uppers[i]))
            axes[i,i].step(bin_edges[:-1], hist, where="post", color="C0")
            axes[i,i].set_ylim(0, hist.max()*1.1)

            if condition_mask[i]:
                #print(x[i])
                axes[i,i].vlines(x[i],0,hist.max()*1.1, colors='r')

        return axes


    def init(figsize=(10,10), labels=None):

        global condition_mask
        global x
        global j
        condition_mask = jnp.array([False]*lowers.shape[0])
        j = 0
        # Precompile
        condition_mask2 = condition_mask
        for i in range(len(x)//2):
            condition_mask2 = condition_mask2.at[i].set(True)
            x_o2 = x[condition_mask2]
            _ = sample_fn(jax.random.PRNGKey(2), condition_mask2, x_o2)

        x_o = x[condition_mask]

        samples1 = sample_fn(jax.random.PRNGKey(0), condition_mask, x_o)


        fig, axes = pairplot(np.array(samples1), figsize=figsize,  limits=list(zip(lowers, uppers)), hist_offdiag={"bins":off_diag_bins}, hist_diag={"bins":diag_bins}, points_colors=["red"], points_offdiag={"markersize":5},samples_labels='_Hidden label', labels=labels, points_labels=[r"$x_o$"], legend=False)

        for i in range(len(x)):
            axes[i,i].get_children()[0].remove()

        axes = update_axes(samples1,axes)

        return fig, axes

    def onclick(event):
        global axes
        global condition_mask
        global x
        global j

        ax = event.inaxes
        removed = False

        if ax is not None:
            x_data, y_data =  ax.transData.inverted().transform((event.x, event.y))
            #print(f"Clicked on {x_data}, {y_data}")
            for i in range(len(axes)):

                if ax is axes[i,i]:
                    for child in axes[i,i].get_children():
                        if isinstance(child,mpl.collections.LineCollection):
                            child.remove()
                            removed = True

                    if removed:
                        condition_mask = condition_mask.at[i].set(False)
                        x_o = x[condition_mask]
                        j += 1
                        samples_new = sample_fn(jax.random.PRNGKey(j), condition_mask, x_o)
                        samples_new = np.nan_to_num(samples_new) # Protect against wierd user input
                        axes = update_axes(samples_new, axes)
                        return




                    condition_mask = condition_mask.at[i].set(True)
                    #print(x_data)
                    x = x.at[i].set(x_data)

                    x_o = x[condition_mask]
                    j += 1
                    samples_new = sample_fn(jax.random.PRNGKey(j), condition_mask, x_o)
                    samples_new = np.nan_to_num(samples_new) # Protect against wierd user input
                    axes = update_axes(samples_new, axes)

                    return

    global axes

    fig, axes = init(figsize, labels)
    fig.canvas.mpl_connect('button_press_event', onclick)
    return fig, axes
