import numpy as np
from tvb.analyzers.fmri_balloon import BalloonModel
from tvb.datatypes.time_series import TimeSeries

from tvb_inversion.parameters import Metric
import logging
log = logging.getLogger(__name__)
import pandas as pd


# this would be more elegant if the bold monitor was part of the backend
# then we can simplify and split this into individual metrics 
class BoldFCDForSBI(Metric): # pack all necessary functions to the metric callable
    def __init__(self, win_len=20, win_sp=1):
        self.win_len = win_len
        self.win_sp  = win_sp
        self.n_vals = 7
        self.summary_stats_labels = [ 'HOMO_FC', 'FCD_STACK_STD_INTER_TENS', 'FCD_SUBJ_DIFF_VAR_OV_TENS' ]
        self.summary_stats_idx = [2,5,6]


    def __call__(self, t, y):
        t *= 10.
        if np.any(np.isnan(y)): 
            log.error("NaN in the simulated time series")
            return np.array([None]*self.n_vals) # there should be a better way
        bold_t, bold_d = tavg_to_bold(t, y, 1.) # TODO check for NaN and skip next steps in that case
        if np.any(np.isnan(bold_d)): 
            log.error("NaN in the simulated BOLD")
            return np.array([None]*self.n_vals) # there should be a better way
        HOMO_FC, FCD, FCD_inter, FCD_STACK_STD_INTER_TENS, FCD_SUBJ_DIFF_VAR_OV_TENS = compute_summary_statistics(bold_d, win_len=self.win_len, win_sp=self.win_sp)
        
        return np.array([bold_t, bold_d, HOMO_FC, FCD, FCD_inter, FCD_STACK_STD_INTER_TENS, FCD_SUBJ_DIFF_VAR_OV_TENS]) # this is abuse


def tavg_to_bold(tavg_t, tavg_d, tavg_period, svar=0, decimate=2000):
    tsr = TimeSeries(
        data = tavg_d[:,[svar],:,:],
        time = tavg_t,
        sample_period = tavg_period
    )
    tsr.configure()
    bold_model = BalloonModel(time_series = tsr, dt=tavg_period/1000)
    bold_data_analyzer  = bold_model.evaluate()
    bold_t = bold_data_analyzer.time[::decimate] * 1000 # to ms
    bold_d = bold_data_analyzer.data[::decimate,:]
    return bold_t, bold_d 


def compute_summary_statistics(bold_d, win_len, win_sp):
    N = bold_d.shape[2]
    NHALF = N//2
    _ , inter_mask = get_masks(N)
    rsFC = compute_fc(bold_d[:,0,:,0].T)        
    HOMO_FC = np.mean(np.diag(rsFC,k=NHALF))
    FCD, _ = compute_fcd(bold_d[:,0,:,0], win_len=win_len, win_sp=win_sp)
    FCD_inter, fc_stack_inter, _ = compute_fcd_filt(bold_d[:,0,:,0], mat_filt=inter_mask, win_len=win_len, win_sp=win_sp )
    FCD_STACK_STD_INTER_TENS  = np.std(fc_stack_inter) # FCD_OSC_OV_INTER_vect
    FCD_VAR_OV_vect           = np.var(np.triu(FCD, k=win_len))
    FCD_VAR_OV_INTER_vect     = np.var(np.triu(FCD_inter, k=win_len))
    FCD_SUBJ_DIFF_VAR_OV_TENS = FCD_VAR_OV_INTER_vect - FCD_VAR_OV_vect # this could be easily done in the final dataframe
    return HOMO_FC, FCD, FCD_inter, FCD_STACK_STD_INTER_TENS, FCD_SUBJ_DIFF_VAR_OV_TENS


def compute_fcd(ts, win_len=30, win_sp=1):
    """
    Arguments:
        ts:      time series of shape [time,nodes]
        win_len: sliding window length in samples
        win_sp:  sliding window step in samples
    Returns:
        FCD: matrix of functional connectivity dynamics
        fcs: windowed functional connectivity matrices
    """
    n_samples, n_nodes = ts.shape
    fc_triu_ids = np.triu_indices(n_nodes, 1)
    n_fcd = len(fc_triu_ids[0])
    fc_stack = []
    for t0 in range( 0, ts.shape[0]-win_len, win_sp ):
        t1=t0+win_len
        fc = np.corrcoef(ts[t0:t1,:].T)
        fc = fc[fc_triu_ids]
        fc_stack.append(fc)
    fcs = np.array(fc_stack)
    FCD = np.corrcoef(fcs)
    return FCD, fcs

def compute_fcd_filt(ts, mat_filt, win_len=30, win_sp=1):
    """Compute dynamic functional connectivity with FC matrix filtering
    Arguments:
        ts:       time series of shape [time,nodes]
        mat_filt: binary matrix to act as the filter [nodes, nodes]
        win_len:  sliding window length in samples
        win_sp:   sliding window step in samples
    Returns:
        FCD: matrix of functional connectivity dynamics
        fcs: windowed functional connectivity matrices
        speed_fcd: rate of changes between FC frames
    """
    n_samples, n_nodes = ts.shape 
    fc_triu_ids = np.triu_indices(n_nodes, 1) #returns the indices for upper triangle
    n_fcd = len(fc_triu_ids[0]) #
    fc_stack    = []
    speed_stack = []
    for t0 in range( 0, ts.shape[0]-win_len, win_sp ):
        t1=t0+win_len
        fc = np.corrcoef(ts[t0:t1,:].T)
        fc = fc*(fc>0)*(mat_filt)
        fc = fc[fc_triu_ids]
        fc_stack.append(fc)
        if t0 > 0 :
            corr_fcd  = np.corrcoef([fc,fc_prev])[0,1]
            speed_fcd = 1-corr_fcd
            speed_stack.append(speed_fcd)
            fc_prev   = fc
        else:
            fc_prev   = fc
            
    fcs      = np.array(fc_stack)
    speed_ts = np.array(speed_stack)
    FCD = np.corrcoef(fcs)
    return FCD, fcs, speed_ts

def compute_fc(bold_d):
    rsFC = np.corrcoef(bold_d)
    rsFC = rsFC * (rsFC>0)
    rsFC = rsFC - np.diag(np.diagonal(rsFC))
    return rsFC


def get_masks(N):
    intra_mask = np.zeros((N,N), dtype=np.bool)
    N = N//2
    intra_mask[:N,:N] = 1
    intra_mask[N:,N:] = 1
    inter_mask = ~intra_mask
    return intra_mask, inter_mask




import tvb.datatypes.time_series as time_series
from tvb.basic.neotraits.api import HasTraits, Attr, NArray, Range, Float
import tvb.simulator.integrators as integrators_module
import numpy
# need to investigate bit more, before that and the subsequent fix, we use
# modified version.
class BalloonModel(HasTraits):
    """
    A class for calculating the simulated BOLD signal given a TimeSeries
    object of TVB and returning another TimeSeries object.
    The haemodynamic model parameters based on constants for a 1.5 T scanner.
    """
    # NOTE: a potential problem when the input is a TimeSeriesSurface.
    # TODO: add an spatial averaging for TimeSeriesSurface.
    time_series = Attr(
        field_type=time_series.TimeSeries,
        label="Time Series",
        required=True,
        doc="""The timeseries that represents the input neural activity""")
    # it also sets the bold sampling period.
    dt = Float(
        label=":math:`dt`",
        default=0.002,
        required=True,
        doc="""The integration time step size for the balloon model (s).
        If none is provided, by default, the TimeSeries sample period is used.""")
    integrator = Attr(
        field_type=integrators_module.Integrator,
        label="Integration scheme",
        default=integrators_module.HeunDeterministic(),
        required=True,
        doc=""" A tvb.simulator.Integrator object which is
        an integration scheme with supporting attributes such as
        integration step size and noise specification for stochastic
        methods. It is used to compute the time courses of the balloon model state
        variables.""")
    bold_model = Attr(
        field_type=str,
        label="Select BOLD model equations",
        choices=("linear", "nonlinear"),
        default="nonlinear",
        doc="""Select the set of equations for the BOLD model.""")
    RBM = Attr(
        field_type=bool,
        label="Revised BOLD Model",
        default=True,
        required=True,
        doc="""Select classical vs revised BOLD model (CBM or RBM).
        Coefficients  k1, k2 and k3 will be derived accordingly.""")
    neural_input_transformation = Attr(
        field_type=str,
        label="Neural input transformation",
        choices=("none", "abs_diff", "sum"),
        default="none",
        doc=""" This represents the operation to perform on the state-variable(s) of
        the model used to generate the input TimeSeries. ``none`` takes the
        first state-variable as neural input; `` abs_diff`` is the absolute
        value of the derivative (first order difference) of the first state variable;
        ``sum``: sum all the state-variables of the input TimeSeries.""")
    tau_s = Float(
        label=r":math:`\tau_s`",
        default=1.54,
        required=True,
        doc="""Balloon model parameter. Time of signal decay (s)""")
    tau_f = Float(
        label=r":math:`\tau_f`",
        default=1.44,
        required=True,
        doc=""" Balloon model parameter. Time of flow-dependent elimination or
        feedback regulation (s). The average  time blood take to traverse the
        venous compartment. It is the  ratio of resting blood volume (V0) to
        resting blood flow (F0).""")
    tau_o = Float(
        label=r":math:`\tau_o`",
        default=0.98,
        required=True,
        doc="""
        Balloon model parameter. Haemodynamic transit time (s). The average
        time blood take to traverse the venous compartment. It is the  ratio
        of resting blood volume (V0) to resting blood flow (F0).""")
    alpha = Float(
        label=r":math:`\tau_f`",
        default=0.32,
        required=True,
        doc="""Balloon model parameter. Stiffness parameter. Grubb's exponent.""")
    TE = Float(
        label=":math:`TE`",
        default=0.04,
        required=True,
        doc="""BOLD parameter. Echo Time""")
    V0 = Float(
        label=":math:`V_0`",
        default=4.0,
        required=True,
        doc="""BOLD parameter. Resting blood volume fraction.""")
    E0 = Float(
        label=":math:`E_0`",
        default=0.4,
        required=True,
        doc="""BOLD parameter. Resting oxygen extraction fraction.""")
    epsilon = NArray(
        label=":math:`\epsilon`",
        default=numpy.array([0.5]),
        domain=Range(lo=0.5, hi=2.0, step=0.25),
        required=True,
        doc=""" BOLD parameter. Ratio of intra- and extravascular signals. In principle  this
        parameter could be derived from empirical data and spatialized.""")
    nu_0 = Float(
        label=r":math:`\nu_0`",
        default=40.3,
        required=True,
        doc="""BOLD parameter. Frequency offset at the outer surface of magnetized vessels (Hz).""")
    r_0 = Float(
        label=":math:`r_0`",
        default=25.,
        required=True,
        doc=""" BOLD parameter. Slope r0 of intravascular relaxation rate (Hz). Only used for
        ``revised`` coefficients. """)
    def evaluate(self):
        """
        Calculate simulated BOLD signal
        """
        cls_attr_name = self.__class__.__name__ + ".time_series"
        # self.time_series.trait["data"].log_debug(owner=cls_attr_name)
        # NOTE: Just using the first state variable, although in the Bold monitor
        #      input is the sum over the state-variables. Only time-series
        #      from basic monitors should be used as inputs.
        neural_activity, t_int = self.input_transformation(self.time_series, self.neural_input_transformation)
        input_shape = neural_activity.shape
        result_shape = self.result_shape(input_shape)
        self.log.debug("Result shape will be: %s" % str(result_shape))
        if self.dt is None:
            self.dt = self.time_series.sample_period / 1000.  # (s) integration time step
            msg = "Integration time step size for the balloon model is %s seconds" % str(self.dt)
            self.log.debug(msg)
        # NOTE: Avoid upsampling ...
        if self.dt < (self.time_series.sample_period / 1000.):
            msg = "Integration time step shouldn't be smaller than the sampling period of the input signal."
            self.log.error(msg)
        balloon_nvar = 4
        # NOTE: hard coded initial conditions
        state = numpy.zeros((input_shape[0], balloon_nvar, input_shape[2], input_shape[3]))  # s
        state[0, 1, :] = 1.  # f
        state[0, 2, :] = 1.  # v
        state[0, 3, :] = 1.  # q
        # BOLD model coefficients
        k = self.compute_derived_parameters()
        k1, k2, k3 = k[0], k[1], k[2]
        # prepare integrator
        self.integrator.dt = self.dt
        self.integrator.configure()
        self.log.debug("Integration time step size will be: %s seconds" % str(self.integrator.dt))
        scheme = self.integrator.scheme
        # NOTE: the following variables are not used in this integration but
        # required due to the way integrators scheme has been defined.
        local_coupling = 0.0
        stimulus = 0.0
        # Do some checks:
        if numpy.isnan(neural_activity).any():
            self.log.warning("NaNs detected in the neural activity!!")
        # normalise the time-series.
        #neural_activity = neural_activity - neural_activity.mean(axis=0)[numpy.newaxis, :]
        # solve equations
        for step in range(1, t_int.shape[0]):
            state[step, :] = scheme(state[step - 1, :], self.balloon_dfun,
                                    neural_activity[step, :], local_coupling, stimulus)
            if numpy.isnan(state[step, :]).any():
                self.log.warning("NaNs detected...")
        # NOTE: just for the sake of clarity, define the variables used in the BOLD model
        s = state[:, 0, :]
        f = state[:, 1, :]
        v = state[:, 2, :]
        q = state[:, 3, :]
        # import pdb; pdb.set_trace()
        # BOLD models
        if self.bold_model == "nonlinear":
            """
            Non-linear BOLD model equations.
            Page 391. Eq. (13) top in [Stephan2007]_
            """
            y_bold = numpy.array(self.V0 * (k1 * (1. - q) + k2 * (1. - q / v) + k3 * (1. - v)))
            y_b = y_bold[:, numpy.newaxis, :, :]
            self.log.debug("Max value: %s" % str(y_b.max()))
        else:
            """
            Linear BOLD model equations.
            Page 391. Eq. (13) bottom in [Stephan2007]_
            """
            y_bold = numpy.array(self.V0 * ((k1 + k2) * (1. - q) + (k3 - k2) * (1. - v)))
            y_b = y_bold[:, numpy.newaxis, :, :]
        sample_period = 1. / self.dt
        bold_signal = time_series.TimeSeries(
            data=y_b,
            time=t_int,
            sample_period=sample_period,
            sample_period_unit='s')
        return bold_signal
    def compute_derived_parameters(self):
        """
        Compute derived parameters :math:`k_1`, :math:`k_2` and :math:`k_3`.
        """
        if not self.RBM:
            """
            Classical BOLD Model Coefficients [Obata2004]_
            Page 389 in [Stephan2007]_, Eq. (3)
            """
            k1 = 7. * self.E0
            k2 = 2. * self.E0
            k3 = 1. - self.epsilon
        else:
            """
            Revised BOLD Model Coefficients.
            Generalized BOLD signal model.
            Page 400 in [Stephan2007]_, Eq. (12)
            """
            k1 = 4.3 * self.nu_0 * self.E0 * self.TE
            k2 = self.epsilon * self.r_0 * self.E0 * self.TE
            k3 = 1 - self.epsilon
        return numpy.array([k1, k2, k3])
    def input_transformation(self, time_series, mode):
        """
        Perform an operation on the input time-series.
        """
        self.log.debug("Computing: %s on the input time series" % str(mode))
        if mode == "none":
            ts = time_series.data[:, 0, :, :]
            ts = ts[:, numpy.newaxis, :, :]
            t_int = time_series.time / 1000.  # (s)
        elif mode == "abs_diff":
            ts = abs(numpy.diff(time_series.data, axis=0))
            t_int = (time_series.time[1:] - time_series.time[0:-1]) / 1000.  # (s)
        elif mode == "sum":
            ts = numpy.sum(time_series.data, axis=1)
            ts = ts[:, numpy.newaxis, :, :]
            t_int = time_series.time / 1000.  # (s)
        else:
            self.log.error("Bad operation/transformation mode, must be one of:")
            self.log.error("('abs_diff', 'sum', 'none')")
            raise Exception("Bad transformation mode")
        return ts, t_int
    def balloon_dfun(self, state_variables, neural_input, local_coupling=0.0):
        r"""
        The Balloon model equations. See Eqs. (4-10) in [Stephan2007]_
        .. math::
                \frac{ds}{dt} &= x - \kappa\,s - \gamma \,(f-1) \\
                \frac{df}{dt} &= s \\
                \frac{dv}{dt} &= \frac{1}{\tau_o} \, (f - v^{1/\alpha})\\
                \frac{dq}{dt} &= \frac{1}{\tau_o}(f \, \frac{1-(1-E_0)^{1/\alpha}}{E_0} - v^{&/\alpha} \frac{q}{v})\\
                \kappa &= \frac{1}{\tau_s}\\
                \gamma &= \frac{1}{\tau_f}
        """
        s = state_variables[0, :]
        f = state_variables[1, :]
        v = state_variables[2, :]
        q = state_variables[3, :]
        x = neural_input[0, :]
        ds = x - (1. / self.tau_s) * s - (1. / self.tau_f) * (f - 1)
        df = s
        dv = (1. / self.tau_o) * (f - v ** (1. / self.alpha))
        dq = (1. / self.tau_o) * ((f * (1. - (1. - self.E0) ** (1. / f)) / self.E0) -
                                  (v ** (1. / self.alpha)) * (q / v))
        return numpy.array([ds, df, dv, dq])
    def result_shape(self, input_shape):
        """Returns the shape of the main result of fmri balloon ..."""
        result_shape = (input_shape[0], input_shape[1],
                        input_shape[2], input_shape[3])
        return result_shape
    def result_size(self, input_shape):
        """
        Returns the storage size in Bytes of the main result of .
        """
        result_size = numpy.sum(list(map(numpy.prod, self.result_shape(input_shape)))) * 8.0  # Bytes
        return result_size
    def extended_result_size(self, input_shape):
        """
        Returns the storage size in Bytes of the extended result of the ....
        That is, it includes storage of the evaluated ... attributes
        such as ..., etc.
        """
        extend_size = self.result_size(input_shape)  # Currently no derived attributes.
        return extend_size
