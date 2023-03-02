from typing import Dict

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


def plot_posterior_samples_model_parameters(dataframe, simulation_params):

    fig = plt.figure(figsize=(15, 10))
    ax = sns.violinplot(data=dataframe, bw=.1)
    plt.setp(ax.collections, alpha=.5)
    label = "ground truth"
    if isinstance(simulation_params, list):
        for i, v in enumerate(simulation_params):
            plt.axhline(v, xmin=i*(1/len(dataframe.columns)),  xmax=(i+1)*(1/len(dataframe.columns)), linestyle="--", linewidth=2, color="black", label=label)
            label = "_nolegend_"
    else:
        plt.axhline(simulation_params, xmin=0.0, linestyle="--", linewidth=2, color="black", label=label)
    plt.title("model_a", fontsize=16)
    plt.tick_params(axis="both", labelsize=16)
    plt.legend(fontsize=16)
    plt.show()


def plot_posterior_samples_global_parameters(dataframe, simulation_params):

    fig, axes = plt.subplots(ncols=1, nrows=len(dataframe.columns), figsize=(len(dataframe.columns)*5, 15))
    label = "ground truth"
    for i, key in enumerate(dataframe.columns):
        ax = axes.reshape(-1)[i]
        sns.violinplot(y=dataframe[key], bw=.1, ax=ax)
        plt.setp(ax.collections, alpha=.5)
        try:
            ax.axhline(simulation_params[key], linestyle="--", linewidth=2, color="black", label=label)
        except KeyError:
            ax.axhline(0.0, linestyle="--", linewidth=2, color="black", label=label)
        if i == 0:
            ax.legend(fontsize=16)
        label = "_nolegend_"
        ax.tick_params(axis="both", labelsize=16)
        ax.set_ylabel(key, size=16)
    plt.show()


def posterior_pairplot(dataframe, size):

    with sns.plotting_context(rc={"axes.labelsize":20}):
        ax = sns.pairplot(data=dataframe, kind="hist", height=size)  # , y_vars=["coupling_a"], x_vars=[k for k, _ in data.items() if "model_a" in k])
    ax.tick_params(axis="both", labelsize=20)
    plt.show()


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
