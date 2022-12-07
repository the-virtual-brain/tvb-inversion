import numpy as np


def zscore(true_mean, post_mean, post_std):
    """
    calculate z-score

    parameters
    ------------
    true_mean: float
        true value of the parameter
    post_mean: float
        mean [max] value of the posterior
    post_std: float
        standard deviation of postorior

    return
    --------

    z-score: float

    """
    return np.abs((post_mean - true_mean) / post_std)


def shrinkage(prior_std, post_std):
    """
    shrinkage = 1 -  \frac{sigma_{post}/sigma_{prior}} ^2

    parameters
    -----------
    prior_std: float
        standard deviation of prior
    post_std: float
        standard deviation of postorior

    return
    ----------
    shrinkage: float

    """
    return 1 - (post_std / prior_std)**2