from typing import Optional, Callable
import pymc as pm
import numpy as np
import pytensor
import pytensor.tensor as pyt
from tvb_inversion.base.gen_model import StatisticalModel
from tvb_inversion.base.observation_models import linear
from tvb_inversion.pymc.prior import PymcPriors, PymcPrior
from tvb_inversion.pymc.stats_model import PymcModel
from tvb.simulator.simulator import Simulator
from tvb.simulator.backend.pytensor import PytensorBackend


class PymcModelBuilder(StatisticalModel):

    def __init__(
            self,
            sim: Simulator,
            params: Optional[PymcPriors] = None,
            model: Optional[pm.Model] = None,
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

        self.dfun: Optional[Callable] = None
        self.cfun: Optional[Callable] = None
        self.ifun: Optional[Callable] = None
        self.mfun: Optional[Callable] = None

    def configure(self):
        assert isinstance(self.params, PymcPriors)
        assert isinstance(self.model, pm.Model)

    def _build_or_append_prior(self, prior):
        if self.params:
            self.params.append(prior)
        else:
            self.params = prior
        return self.params

    def build_dfun(self):
        template = """
           import pytensor
           import pytensor.tensor as pyt
           import numpy as np
           <%include file="pytensor-dfuns.py.mako"/>
           """

        return PytensorBackend().build_py_func(
            template_source=template, content=dict(sim=self.sim, mparams=list(self.params.get_model_params().keys())), name="dfuns", print_source=False)

    def build_cfun(self):
        template = f"""
           import pytensor
           import pytensor.tensor as pyt
           import numpy as np
           n_node = {self.sim.connectivity.number_of_regions}
           <%include file="pytensor-coupling.py.mako"/>
           """

        return PytensorBackend().build_py_func(
            template_source=template, content=dict(sim=self.sim, cparams=list(self.params.get_coupling_params().keys())), name="coupling", print_source=False)

    def build_ifun(self):
        template = """
            import pytensor
            import pytensor.tensor as pyt
            import numpy as np
            <%include file="pytensor-integrate.py.mako"/>
            """
        return PytensorBackend().build_py_func(
            template_source=template, content=dict(sim=self.sim, np=np, pyt=pyt), name='integrate', print_source=False)

    def build_mfun(self):
        pass

    def build_ofun(self, x_sim):
        with self.model:
            x_hat = pm.Deterministic(
                name="x_hat", var=self.obs_fun(x_sim, **self.params.get_observation_model_params()))
        return x_hat
    
    def build_sim(self, x_init, **kwargs):

        template = '<%include file="pytensor-sim.py.mako"/>'
        content = dict(sim=self.sim, mparams=list(self.params.get_model_params().keys()), cparams=list(self.params.get_coupling_params().keys()), np=np, pyt=pyt)
        kernel, _ = PytensorBackend().build_py_func(template, content, name="kernel,default_noise", print_source=False)

        if not "idelays" in kwargs:
            if self.sim.connectivity.idelays.any():
                kwargs["idelays"] = self.sim.connectivity.delay_indices

        with self.model:
            x_sim = kernel(
                state=x_init,
                weights=self.sim.connectivity.weights,
                trace=pyt.zeros((int(self.sim.simulation_length/self.sim.integrator.dt),) + self.sim.initial_conditions[:, :, :, 0].shape),
                parmat=self.sim.model.spatial_parameter_matrix,
                mparams=self.prior.get_model_params(),
                cparams=self.prior.get_coupling_params(),
                **kwargs
            )

            # x_sim = pytensor.scan(
            #     fn=self.scheme,
            #     sequences=kwargs.get("sequence", None),
            #     outputs_info=[x_init[-1]],
            #     non_sequences=list(self.params.get_model_params().values()) + list(self.params.get_coupling_params().values()),
            #     n_steps=self.n_steps
            # )
        
        return x_sim

    # def scheme(self, x_prev, *params):
    #     state = pyt.zeros((self.sim.connectivity.horizon, self.sim.model.nvar, self.sim.connectivity.number_of_regions))
    #     state = pyt.set_subtensor(state[0], x_prev)
    #     state = pyt.transpose(state, axes=[1, 0, 2])
    #
    #     cX = pyt.zeros((self.sim.history.n_cvar, self.sim.history.n_node))
    #     cX = self.cfun(cX, self.sim.connectivity.weights, state, self.sim.connectivity.delay_indices,
    #                    **self.params.get_coupling_params())
    #
    #     dX = pyt.zeros((self.sim.model.nvar, self.sim.history.n_node))
    #     dX = self.dfun(dX, x_prev, cX, self.sim.model.spatial_parameter_matrix, **self.params.get_model_params())
    #
    #     return self.build_ifun(x_prev, dX)

    def build_initial_conditions(self):
        # Get initial conditions from simulator.initial_conditions
        # The history buffer cannot do the job because it holds only cvars
        # The observation cannot do the job either because it might not exist at this stage
        if "x_init" in self.params.names:
            return self.params.dict['x_init']
        return pyt.as_tensor_variable(self.sim.initial_conditions[:, :, :, 0], name="x_init")

    def build_funs(self):
        self.dfun = self.build_dfun()
        self.cfun = self.build_cfun()
        self.ifun = self.build_ifun()
        # self.mfun: self.build_mfun()

    def compose_model(self):

        self.build_funs()

        with self.model:
            x_sim = self.build_sim(self.build_initial_conditions())

            if self.obs_fun:
                x_hat = self.build_ofun(x_sim)
            else:
                x_hat = x_sim

            if self.obs is not None:
                x_obs = pm.Normal(name="x_obs", mu=x_hat[:, self.sim.model.cvar, 0, :], sigma=self.params.dict.get("observation_noise", 1.0),
                                  shape=self.obs.shape, observed=self.obs)

    def build(self):
        return PymcModel(self.sim, self.params)


class DeterministicPymcModelBuilder(PymcModelBuilder):
    pass


class StochasticPymcModelBuilder(DeterministicPymcModelBuilder):

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
            nsig = self.params.dict.get("integrator.noise.nsig", self.sim.integrator.noise.nsig[0].item())
            if "dWt_star" in self.params.dict:
                dWt = pm.Deterministic(name="dWt",
                                       var=pyt.sqrt(2.0 * nsig * self.sim.integrator.dt) * self.params.dict["dWt_star"])
            else:
                dWt_star = pm.Normal(name="dWt_star", mu=0.0, sigma=1.0, shape=(self.obs.shape[0], *self.sim.initial_conditions.shape[1:-1]))
                dWt = pm.Deterministic(name="dWt",
                                       var=pyt.sqrt(2.0 * nsig * self.sim.integrator.dt) * dWt_star)

        return dWt

    def build_funs(self):
        super().build_funs()
        self.nfun = self.build_nfun()

    def build_sim(self, x_init, dWt=None, **kwargs):
        if dWt is None:
            dWt = self.build_nfun()
        return super().build_sim(x_init, noise=dWt, **kwargs)

    # def scheme(self, dWt, x_prev, *params):
    #     return super().scheme(x_prev, *params) + dWt


class DefaultDeterministicPymcModelBuilder(DeterministicPymcModelBuilder):

    def __init__(
            self,
            sim: Simulator,
            params: Optional[PymcPrior] = None,
            model: Optional[pm.Model] = None,
            observation: Optional[np.ndarray] = None,
            n_steps: Optional[int] = None
    ):
        super().__init__(sim, params=params, model=model,
                         observation=observation, observation_fun=linear, n_steps=n_steps)

    def set_initial_conditions(self, def_std=0.1):
        with self.model:
            x_init_star = pm.Normal(name="x_init_star", mu=0.0, sigma=1.0,
                                    shape=self.sim.initial_conditions.shape[:-1])
            x_init = pm.Deterministic(name="x_init",
                                      var=self.sim.initial_conditions[:, :, :, 0] * (1.0 + def_std * x_init_star))

        return x_init

    def set_observation_model(self, def_std=0.1):
        with self.model:
            amplitude_star = pm.Normal(name="amplitude_star", mu=0.0, sigma=1.0)
            amplitude = pm.Deterministic(name="amplitude", var=1.0 + def_std * amplitude_star)

            offset_star = pm.Normal(name="offset_star", mu=0.0, sigma=1.0)
            offset = pm.Deterministic(name="offset", var=def_std * offset_star)

            observation_noise_star = pm.HalfNormal(name="observation_noise_star", sigma=1.0)
            observation_noise = pm.Deterministic(name="observation_noise", var=def_std * observation_noise_star)

        return amplitude, offset, observation_noise

    def _set_default_priors(self, def_std=0.1):
        x_init = self.set_initial_conditions(def_std)
        amplitude, offset, observation_noise = self.set_observation_model(def_std)
        return x_init, amplitude, offset, observation_noise

    def set_default_priors(self, def_std=0.1):
        x_init, amplitude, offset, observation_noise = self._set_default_priors(def_std)
        names = ["x_init",
                 "observation.amplitude", "observation.offset", "observation_noise"]
        dists = [x_init,
                amplitude, offset, observation_noise]
        for name, dist in zip(names, dists):
            self.params = self._build_or_append_prior(PymcPrior(self.model, name, dist))
        return self.params


class DefaultStochasticPymcModelBuilder(StochasticPymcModelBuilder, DefaultDeterministicPymcModelBuilder):

    def set_noise(self, def_std=0.1):
        with self.model:
            nsig_star = pm.Normal(name="nsig_star", mu=0.0, sigma=1.0)
            nsig = pm.Deterministic(name="nsig", var=self.sim.integrator.noise.nsig[0] * (1.0 + def_std * nsig_star))
            dWt_star = pm.Normal(name="dWt_star", mu=0.0, sigma=1.0, shape=(self.obs.shape[0], *self.sim.initial_conditions.shape[1:-1]))

        return nsig, dWt_star

    def _set_default_priors(self, def_std=0.1):
        x_init_offset, amplitude, offset, observation_noise = super()._set_default_priors(def_std)
        nsig, dWt_star = self.set_noise(def_std)
        return x_init_offset, nsig, dWt_star, amplitude, offset, observation_noise

    def set_default_priors(self, def_std=0.1):
        x_init, nsig, dWt_star, amplitude, offset, observation_noise = self._set_default_priors(def_std)
        names = ["x_init", "integrator.noise.nsig", "dWt_star",
                 "observation.amplitude", "observation.offset", "observation_noise"]
        dists = [x_init, nsig, dWt_star,
                amplitude, offset, observation_noise]
        for name, dist in zip(names, dists):
            self.params = self._build_or_append_prior(PymcPrior(self.model, name, dist))
        return self.params