import numpy as np
from scipy.stats import kurtosis
from scipy.stats import moment
from scipy.stats import skew
from sbi_tvb import analysis


def _calculate_summary_statistics(self, x, features=None):
    """
    Calculate summary statistics via numpy and scipy.
    The function here extracts 10 momenta for each bold channel, FC mean, FCD mean, variance
    difference and standard deviation of FC stream.
    Check that you can compute FCD features via proper FCD packages

    Parameters
    ----------
    x : output of the simulator

    Returns
    -------
    np.array, summary statistics
    """
    if features is None:
        features = ['higher_moments', 'FC_corr', 'FCD_corr']
    nn = self.simulator.connectivity.weights.shape[0]

    X = x.reshape(nn, int(x.shape[0] / nn))

    n_summary = 16 * nn + (nn * nn) + 300 * 300
    bold_dt = 2250

    sum_stats_vec = np.concatenate((np.mean(X, axis=1),
                                    np.median(X, axis=1),
                                    np.std(X, axis=1),
                                    skew(X, axis=1),
                                    kurtosis(X, axis=1),
                                    ))

    for item in features:

        if item == 'higher_moments':
            sum_stats_vec = np.concatenate((sum_stats_vec,
                                            moment(X, moment=2, axis=1),
                                            moment(X, moment=3, axis=1),
                                            moment(X, moment=4, axis=1),
                                            moment(X, moment=5, axis=1),
                                            moment(X, moment=6, axis=1),
                                            moment(X, moment=7, axis=1),
                                            moment(X, moment=8, axis=1),
                                            moment(X, moment=9, axis=1),
                                            moment(X, moment=10, axis=1),
                                            ))

        if item == 'FC_corr':
            FC = np.corrcoef(X)
            off_diag_sum_FC = np.sum(FC) - np.trace(FC)
            print('FC_Corr')
            sum_stats_vec = np.concatenate((sum_stats_vec,
                                            np.array([off_diag_sum_FC]),
                                            ))

        if item == 'FCD_corr':
            win_FCD = 40e3
            NHALF = int(nn / 2)

            mask_inter = np.zeros([nn, nn])
            mask_inter[0:NHALF, NHALF:NHALF * 2] = 1
            mask_inter[NHALF:NHALF * 2, 0:NHALF] = 1

            bold_summ_stat = X.T

            FCD, fc_stack, speed_fcd = analysis.compute_fcd(bold_summ_stat, win_len=int(win_FCD / bold_dt),
                                                            win_sp=1)
            fcd_inter, fc_stack_inter, _ = analysis.compute_fcd_filt(bold_summ_stat, mask_inter,
                                                                     win_len=int(win_FCD / bold_dt), win_sp=1)

            FCD_TRIU = np.triu(FCD, k=1)

            FCD_INTER_TRIU = np.triu(fcd_inter, k=1)

            FCD_MEAN = np.mean(FCD_TRIU)
            FCD_VAR = np.var(FCD_TRIU)
            FCD_OSC = np.std(fc_stack)
            FCD_OSC_INTER = np.std(fc_stack_inter)

            FCD_MEAN_INTER = np.mean(FCD_INTER_TRIU)
            FCD_VAR_INTER = np.var(FCD_INTER_TRIU)

            DIFF_VAR = FCD_VAR_INTER - FCD_VAR

            sum_stats_vec = np.concatenate((sum_stats_vec,
                                            np.array([FCD_MEAN]), np.array([FCD_OSC_INTER]), np.array([DIFF_VAR])
                                            ))

    sum_stats_vec = sum_stats_vec[0:n_summary]

    return sum_stats_vec
