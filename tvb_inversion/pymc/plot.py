"""This module contains plotting functions for pymc results. However, they are little general and very problem specific.
Generic plotting functions will implemented later."""
from typing import Dict

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

def plot_posterior_samples(inference_data, init_params: Dict[str, float], save: bool = False):
    num_params = len(init_params)
    nrows = int(np.ceil(np.sqrt(num_params)))
    ncols = int(np.ceil(num_params / nrows))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 16))
    for ax in axes.reshape(-1):
        ax.set_axis_off()
    for i, (key, value) in enumerate(init_params.items()):
        ax = axes.reshape(-1)[i]
        ax.set_axis_on()

        if "[" in key:
            index = [int(c) for c in key if c.isdigit()][0]
            key_raw = key[:-3]
            posterior_ = inference_data.posterior[key_raw].values[..., index].flatten()
        else:
            posterior_ = inference_data.posterior[key].values.reshape((inference_data.posterior[key].values.size,))

        ax.hist(posterior_, bins=100, alpha=0.5)
        ax.axvline(posterior_.mean(), color="k", label="posterior mean")
        ax.axvline(init_params[key], color="r", label="simulation parameter")
        ax.set_title(key, fontsize=18)
        ax.tick_params(axis="both", labelsize=16)
    try:
        axes[0, 0].legend(fontsize=18)
    except IndexError:
        axes[0].legend(fontsize=18)


# DEPRECATED
def plot_posterior_samples_model_parameters(dataframe, simulation_params, save_fig=None, index=None):

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(15, 10))
    sns.violinplot(data=dataframe, bw=.1, ax=ax)
    plt.setp(ax.collections, alpha=.5)
    label = "ground truth"
    if isinstance(simulation_params, list):
        for i, v in enumerate(simulation_params):
            plt.axhline(v, xmin=i*(1/len(dataframe.columns)),  xmax=(i+1)*(1/len(dataframe.columns)), linestyle="--", linewidth=2, color="black", label=label)
            label = "_nolegend_"
    else:
        plt.axhline(simulation_params, xmin=0.0, linestyle="--", linewidth=2, color="black", label=label)
    plt.ylabel(r"$a_{model}$", fontsize=18)
    plt.tick_params(axis="both", labelsize=18)
    plt.legend(fontsize=18)

    if save_fig:
        if index:
            plt.savefig(save_fig + f"posterior_samples_model_parameters-{index}.png", dpi=600, bbox_inches="tight")
        else:
            plt.savefig(save_fig + "posterior_samples_model_parameters.png", dpi=600, bbox_inches="tight")


def plot_posterior_samples_global_parameters(dataframe, simulation_params, save_fig=None):

    fig, axes = plt.subplots(ncols=1, nrows=len(dataframe.columns), figsize=(15, len(dataframe.columns) * 5))
    label = "ground truth"
    for i, key in enumerate(dataframe.columns):
        try:
            ax = axes.reshape(-1)[i]
        except AttributeError:
            ax = axes
        sns.violinplot(y=dataframe[key], bw=.1, ax=ax)
        plt.setp(ax.collections, alpha=.5)
        try:
            ax.axhline(simulation_params[key], linestyle="--", linewidth=2, color="black", label=label)
        except KeyError:
            ax.axhline(0.0, linestyle="--", linewidth=2, color="black", label=label)
        if i == 0:
            ax.legend(fontsize=18)
        label = "_nolegend_"
        ax.tick_params(axis="both", labelsize=18)

        if key == "coupling_a":
            ylabel = r"$a_{coupling}$"
        elif key == "amplitude":
            ylabel = r"$m_{obs}$"
        elif key == "offset":
            ylabel = r"$n_{obs}$"
        elif key == "measurement_noise":
            ylabel = r"$v_{obs}$"
        else:
            ylabel = key
        ax.set_ylabel(ylabel, size=18)

    if save_fig:
        plt.savefig(save_fig + "posterior_samples_global_parameters.png", dpi=600, bbox_inches="tight")


def posterior_pairplot(dataframe, size, save_fig=None):

    with sns.plotting_context(rc={"axes.labelsize": 40}):
        ax = sns.pairplot(data=dataframe, kind="hist", height=size, aspect=1, corner=True)  # , y_vars=["coupling_a"], x_vars=[k for k, _ in data.items() if "model_a" in k])
    ax.tick_params(axis="both", labelsize=36)

    if save_fig:
        plt.savefig(save_fig + "pairplot.png", dpi=600, bbox_inches="tight")


