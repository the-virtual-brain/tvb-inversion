import pymc3 as pm
from tvb_inversion.base.stats_model import StatisticalModel
from tvb_inversion.base.observation_models import linear
from tvb_inversion.pymc3.prior import Pymc3Prior
from tvb_inversion.pymc3.stats_model import Pymc3Model
from tvb.simulator.simulator import Simulator


class Pymc3ModelBuilder:

    def __init__(
            self,
            sim: Simulator,
            params: Optional[Pymc3Prior]=None,
            model: Optional[pm.Model]=None,
            observation_fun: Optional[Callable] = None,
            observation: Optional[np.ndarray] = None,
            n_steps: Optional[int] = None
    ):
        super().__init__(sim, params)

        if self.params:
            self.model = params.model
        else:
            self.model = model

        self.obs_fun = observation_fun
        self.obs = observation

        self.n_steps = n_steps
        if self.n_steps is None:
            assert isinstance(self.obs, np.ndarray), "Either observation or n_steps has to be given as input arguments!"
            self.n_steps = self.obs.shape[0]

    def configure(self):
        assert isinstance(self.params, Pymc3Prior)
        assert isinstance(self.model, pm.Model)

    def _build_or_append_prior(self, names, dist):
        if self.params:
            self.params.append(names, dist)
        else:
            self.params = Pymc3Prior(names=names, dist=dist)
        return self.params

    def build_dfun(self):
        template = """
           import theano
           import theano.tensor as tt
           import numpy as np
           <%include file="theano-dfuns.py.mako"/>
           """

        return TheanoBackend().build_py_func(template_source=template, content=dict(sim=self.sim), name="dfuns",
                                             print_source=False)

    def build_cfun(self):
        template = f"""
           import theano
           import theano.tensor as tt
           import numpy as np
           n_node = {self.sim.connectivity.number_of_regions}
           <%include file="theano-coupling.py.mako"/>
           """

        return TheanoBackend().build_py_func(template_source=template, content=dict(sim=self.sim), name="coupling",
                                             print_source=False)

    def build_ifun(self, x_prev, dX):
        return x_prev[0] + self.sim.integrator.dt * dX

    def build_mfun(self):
        pass

    def build_ofun(self, x_sim):
        with self.model:
            x_hat = pm.Deterministic(name="x_hat",
                                     var=self.obs_fun(x_sim, **self.params.get_observation_model_params()))
        return x_hat

    def scheme(self, *x_prev):
        x_prev = x_prev[::-1]

        state = tt.stack(x_prev, axis=0)
        state = tt.transpose(state, axes=[1, 0, 2])

        cX = tt.zeros((self.sim.history.n_cvar, self.sim.history.n_node))
        cX = self.cfun(cX, self.sim.connectivity.weights, state, self.sim.connectivity.delay_indices,
                       **self.params.get_coupling_params())

        dX = tt.zeros((self.sim.model.nvar, self.sim.history.n_node))
        dX = self.dfun(dX, x_prev[0], cX, self.sim.model.spatial_parameter_matrix, **self.params.get_model_params)

        return self.build_ifun(x_prev, dX)

    def build_initial_conditions(self):
        # Get initial conditions from simulator.initial_conditions
        # The history buffer cannot do the job because it holds only cvars
        # The observation cannot do the job either because it might not exist at this stage
        if "x_init" in self.params.names:
            return self.params.dict['x_init']
        x_init_sim = tt.shared(self.sim.initial_conditions[:-1])
        if "x_init_offset" in self.params.names:
            with self.model:
                x_init = pm.Deterministic(name='x_init', var=x_init_sim + self.params.dict['x_init_offset'])
            return x_init
        return x_init_sim

    def build_loop(self, x_init, **kwargs):

        with self.model:
            taps = list(-1 * np.arange(self.sim.connectivity.idelays.max() + 1) - 1)[::-1]
            x_sim, updates = theano.scan(
                    fn=self.scheme,
                    sequences=kwargs.get("sequence", None),
                    outputs_info=[dict(initial=x_init, taps=taps)],
                    n_steps=self.n_steps
                )
        return x_sim, updates

    def build_funs(self):
        self.dfun: Callable = self.build_dfun()
        self.cfun: Callable = self.build_cfun()
        # self.ifun: Callable = self.build_ifun()
        # self.mfun: Callable = self.build_mfun()

    def build_model(self):

        self.build_funs()

        with self.model:
            x_sim, updates = self.build_loop(self.build_initial_conditions())

            if self.obs_fun:
                x_hat = self.build_ofun(x_sim)
            else:
                x_hat = x_sim

            if self.obs:
                x_obs = pm.Normal(name="x_obs", mu=x_hat, sd=self.prior.dict.get("observation.noise", 1.0),
                                  shape=self.obs.shape, observed=self.obs)

        return self.model

    def build(self):
        return Pymc3Model(self.sim, self.params)


class DeterministicPymc3ModelBuilder(Pymc3ModelBuilder):

    pass


class StochasticPymc3ModelBuilder(DeterministicPymc3ModelBuilder):

    def build_nfun(self):
        # TODO: Implement this with theano_backend for Additive white noise!:
        # https://github.com/the-virtual-brain/tvb-root/blob/master/tvb_library/tvb/simulator/noise.py
        # noise generation for white noise:
        # def white(self, shape):
        #     "Generate white noise."
        #     noise = numpy.sqrt(self.dt) * self.random_stream.normal(size=shape)
        #     return noise
        # Additive noise gfun:
        # def gfun(self, state_variables):
        #     r"""
        #     Linear additive noise, thus it ignores the state_variables.
        #     .. math::
        #         g(x) = \sqrt{2D}
        #     """
        #     g_x = numpy.sqrt(2.0 * self.nsig)
        #     return g_x
        with self.model:
            # nsig might be a parameter or to be taken from the simulator
            # TODO: broadcast for different shapes of nsig!!!
            nsig = self.params.dict.get('integrator.noise.nsig', self.sim.integrator.nsig[0].item())
            dXt = pm.Deterministic(name='dXt',
                                   var=tt.sqrt(2.0 * nsig * self.sim.integrator.dt) * self.params.dict['dXt_star'])
        return dXt

    # def build_funs(self):
    #     super().build_funs()
    #     # self.nfun: Callable = self.build_nfun()

    def build_loop(self, x_init, **kwargs):
        sequence = kwargs.pop("sequence", [])
        sequence.append(self.build_nfun())
        return super().build_loop(x_init, sequence=sequence, **kwargs)

    def scheme(self, dXt, *x_prev):
        return super().scheme(*xprev) + dXt


class DefaultDeterministicPymc3ModelBuilder(DeterministicPymc3ModelBuilder):

    def __init__(
            self,
            sim: Simulator,
            params: Optional[Pymc3Prior]=None,
            model: Optional[pm.Model]=None,
            observation: Optional[np.ndarray] = None,
            n_steps: Optional[int] = None
    ):

        super().__init__(sim, params=params, model=model,
                         observation=linear, observation_fun=observation, n_steps=n_steps)

    def set_initial_conditions(self, def_std=0.1):

        with self.model:
            x_init_offset_star = pm.Normal(name="x_init_offset_star", mu=0.0, sd=1.0,
                                           shape=self.sim.initial_conditions.shape)
            x_init_offset = pm.Deterministic(name="x_init_offset", var=def_std * x_init_offset_star)

        return x_init_offset

    def set_observation_model(self, def_std=0.1):

        with self.model:
            amplitude_star = pm.Normal(name="amplitude_star", mu=0.0, sd=1.0)
            amplitude = pm.Deterministic(name="observation_model_amplitude", var=1.0 + def_std * amplitude_star)

            offset_star = pm.Normal(name="offset_star", mu=0.0, sd=1.0)
            offset = pm.Deterministic(name="observation_model_offset", var=def_std * offset_star)

            observation_noise_star = pm.HalfNormal(name="observation_noise_star", sigma=1.0)
            observation_noise = pm.Deterministic(name="observation_noise", var=def_std * observation_noise_star)

        return amplitude, offset, observation_noise

    def _set_default_priors(self, def_std=0.1):
        x_init_offset = self.set_initial_conditions(def_std)
        amplitude, offset, observation_noise = self.set_observation_model(def_std)
        return x_init_offset, amplitude, offset, observation_noise

    def set_default_prior(self, def_std=0.1):
        x_init_offset, amplitude, offset, observation_noise = self._set_default_priors(def_std)
        names = ["x_init_offset",
                 "observation.model.amplitude", "observation.model.offset", "observation.noise"],
        dist = [x_init_offset,
                amplitude, offset, observation_noise]
        self.params = self._build_or_append_prior(names, dist)
        return self.params


class DefaultStochasticPymc3ModelBuilder(StochasticPymc3ModelBuilder, DefaultDeterministicPymc3ModelBuilder):

    def set_noise(self, def_std=0.1):

        with self.model:
            nsig_star = pm.HalfNormal(name="nsig_star", sigma=1.0)
            nsig = pm.Deterministic(name="integrator.noise.nsig", var=def_std * nsig_star)
            dXt_star = pm.Normal(name="dX_star", mu=0.0, sd=1.0, shape=x_shape)

        return nsig, dXt_star

    def _set_default_priors(self, def_std=0.1):
        x_init_offset, amplitude, offset, observation_noise = super()._set_default_priors(def_std)
        nsig, dXt_star = self.set_noise(def_std)
        return x_init_offset, nsig, dXt_star, amplitude, offset, observation_noise

    def set_default_prior(self, def_std=0.1):
        x_init_offset, nsig, dXt_star, amplitude, offset, observation_noise = self._set_default_priors(def_std)
        names = ["x_init_offset", "integrator.noise.nsig", "dXt_star",
                 "observation.model.amplitude", "observation.model.offset", "observation.noise"],
        dist = [x_init_offset, nsig, dXt_star,
                amplitude, offset, observation_noise]
        self.params = self._build_or_append_prior(names, dist)
        return self.params
