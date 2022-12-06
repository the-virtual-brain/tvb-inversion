from typing import Dict

import numpy as np
from matplotlib import pyplot as plt


def plot_posterior_samples(inference_data, init_params: Dict[str, float], save: bool = False):
    num_params = len(init_params)
    nrows = int(np.ceil(np.sqrt(num_params)))
    ncols = int(np.ceil(num_params / nrows))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 16))
    for ax in axes.reshape(-1):
        ax.set_axis_off()
    for i, (key, value) in enumerate(init_params.items()):

        posterior_ = inference_data.posterior[key].values.reshape((inference_data.posterior[key].values.size,))
        ax = axes.reshape(-1)[i]
        ax.set_axis_on()
        ax.hist(posterior_, bins=100, alpha=0.5)
        ax.axvline(init_params[key], color="r", label="simulation parameter")
        ax.axvline(posterior_.mean(), color="k", label="posterior mean")

        ax.set_title(key, fontsize=18)
        ax.tick_params(axis="both", labelsize=16)
    try:
        axes[0, 0].legend(fontsize=18)
    except IndexError:
        axes[0].legend(fontsize=18)
