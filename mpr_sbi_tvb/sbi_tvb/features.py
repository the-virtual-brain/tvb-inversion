from enum import Enum
import numpy as np
from scipy.stats import kurtosis
from scipy.stats import moment
from scipy.stats import skew

from sbi_tvb import analysis
from sbi_tvb.logger.builder import get_logger


class FeaturesEnum(Enum):
    HIGHER_MOMENTS = 'higher_moments'
    FC_CORR = 'fc_corr'
    FCD_CORR = 'fcd_corr'

    @staticmethod
    def get_all():
        return [el.name for el in FeaturesEnum]


class SummaryStatistics(object):

    def __init__(self, x, nr_regions, bold_dt=2250):
        self.logger = get_logger(self.__class__.__module__)
        self.nn = nr_regions
        self.bold_dt = bold_dt
        self.X = x.reshape(self.nn, int(x.shape[0] / self.nn))
        self.n_summary = 16 * self.nn + (self.nn * self.nn) + 300 * 300

    def compute(self, features=None):
        """
        Calculate summary statistics via numpy and scipy.
        The function here extracts 10 momenta for each bold channel, FC mean, FCD mean, variance
        difference and standard deviation of FC stream.
        Check that you can compute FCD features via proper FCD packages

        Input
        ----------
        x : output of the simulator

        Returns
        -------
        np.array, summary statistics
        """
        if features is None:
            features = FeaturesEnum.get_all()

        sum_stats_vec = np.concatenate((np.mean(self.X, axis=1),
                                        np.median(self.X, axis=1),
                                        np.std(self.X, axis=1),
                                        skew(self.X, axis=1),
                                        kurtosis(self.X, axis=1),
                                        ))

        for feature in features:
            method_name = FeaturesEnum[feature].value
            method = self.__getattribute__(method_name)
            sum_stats_vec = method(sum_stats_vec)

        sum_stats_vec = sum_stats_vec[0:self.n_summary]

        return sum_stats_vec

    def higher_moments(self, sum_stats_vec):
        self.logger.info('Computing HIGHER_MOMENTS feature...')

        sum_stats_vec = np.concatenate((sum_stats_vec,
                                        moment(self.X, moment=2, axis=1),
                                        moment(self.X, moment=3, axis=1),
                                        moment(self.X, moment=4, axis=1),
                                        moment(self.X, moment=5, axis=1),
                                        moment(self.X, moment=6, axis=1),
                                        moment(self.X, moment=7, axis=1),
                                        moment(self.X, moment=8, axis=1),
                                        moment(self.X, moment=9, axis=1),
                                        moment(self.X, moment=10, axis=1),
                                        ))
        return sum_stats_vec

    def fc_corr(self, sum_stats_vec):
        self.logger.info('Computing FC_CORR feature...')

        FC = np.corrcoef(self.X)
        off_diag_sum_FC = np.sum(FC) - np.trace(FC)
        sum_stats_vec = np.concatenate((sum_stats_vec,
                                        np.array([off_diag_sum_FC]),
                                        ))

        return sum_stats_vec

    def fcd_corr(self, sum_stats_vec):
        self.logger.info('Computing FCD_CORR feature...')

        win_FCD = 40e3
        NHALF = int(self.nn / 2)

        mask_inter = np.zeros([self.nn, self.nn])
        mask_inter[0:NHALF, NHALF:NHALF * 2] = 1
        mask_inter[NHALF:NHALF * 2, 0:NHALF] = 1

        bold_summ_stat = self.X.T

        FCD, fc_stack, speed_fcd = analysis.compute_fcd(bold_summ_stat, win_len=int(win_FCD / self.bold_dt),
                                                        win_sp=1)
        fcd_inter, fc_stack_inter, _ = analysis.compute_fcd_filt(bold_summ_stat, mask_inter,
                                                                 win_len=int(win_FCD / self.bold_dt), win_sp=1)

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
        return sum_stats_vec
