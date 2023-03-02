import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

def plot_posterior_samples_model_parameters(dataframe, simulation_params):

    fig = plt.figure(figsize=(15, 10))
    ax = sns.violinplot(data=dataframe, bw=.1)
    plt.setp(ax.collections, alpha=.5)
    label = "ground truth"
    for i, v in enumerate(simulation_params):
        plt.axhline(v, xmin=i*(1/len(dataframe.columns)),  xmax=(i+1)*(1/len(dataframe.columns)), linestyle="--", linewidth=2, color="black", label=label)
        label = "_nolegend_"
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
        ax.axhline(simulation_params[key], linestyle="--", linewidth=2, color="black", label=label)
        if i == 0:
            ax.legend(fontsize=16)
        label = "_nolegend_"
        ax.tick_params(axis="both", labelsize=16)
        ax.set_ylabel(key, size=16)
    plt.show()
    