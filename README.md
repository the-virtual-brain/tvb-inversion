# TVB Inversion

The `tvb_inversion` package implements the machinery required to perform inference over parameter of The Virtual Brain simulator parameters with the help of Simulation Based Inference (SBI).

## Installation

To install the latest version from the GitHub repository type the following in the terminal:

```shell
$ git clone https://github.com/the-virtual-brain/tvb-inversion.git
$ cd tvb-inversion/
$ pip install .
```

The main dependencies are the [`tvb-library`](https://thevirtualbrain.org/tvb/zwei), [`sbi`](https://github.com/mackelab/sbi), and [Pytorch](https://pytorch.org/).

## Usage

A fully elaborated example is available in the following notebooks:

* `01_config_and_sample.ipynb` -  configuration of the simulation, definition of the priors, sampling
* `02_inference.ipynb` - inference steps, result assessment (diagnostics)

## API

The package is organized around three main steps of the inference: 

1. simulation configuration and definition of priors
2. sampling from priors, running simulations
3. inference and diagnostics

### Simulation configuration and priors

Using TVB `Simulator` instance we define the simulator template which will be used in the inference. 

```python
from tvb.simulator.lab import *

conn = connectivity.Connectivity.from_file()

sim = simulator.Simulator(
    model=models.MontbrioPazoRoxin(),
    connectivity=conn,
    coupling=coupling.Linear(
        a=np.r_[2.45]
    ),
    conduction_speed=conn.speed.item(),
    integrator=integrators.HeunStochastic(
        dt=0.01,
        noise=noise.Additive(
            nsig=np.r_[0.035, 0.035 * 2],
            noise_seed=42
        )
    ),
    monitors=[monitors.TemporalAverage(period=0.1)],
    simulation_length=30e3
)
```

Next, a prior distribution is defined using the `Prior` class. The dimensions of the prior points to attributes in the simulator instance, and the distribution is given by any class supporting the `.sample` and `.log_prob()` methods (e.g. the distributions from PyTorch). Here a one-dimensional uniform prior is given for the coupling strength parameter:

```python
from tvb_inversion.sbi.prior import PytorchPrior
import torch

prior = PytorchPrior(names=['coupling.a'], dist=torch.distributions.Uniform(0.1, 1.2))
```

From the simulator and the prior we create statistical model:
```python
from tvb_inversion.sbi.sbi_model import sbiModel

sbi_model = sbiModel(sim, prior)
```

And lastly, the summary statistics have to be defined - a function which converts the outputs of the simulator to a list of scalar data features. See the `demo.py` for an example.

### Sampling, running simulations

This step makes use of the `parameters` API allowing to configure and perform large parameter sweeps with TVB.  We start with drawing a sample from the prior using the `generate_sim_seq` method. This returns an instance of  `parameters.SimSeq` - simulation sequence comprised of template simulator and the list of parameter value combinations. 

```python
seq = sbi_model.generate_sim_seq(4000) # sample 4000 values from the prior
```

The summary statistics are then wrapped in a single callable following the `parameters.Metric` signature, that is taking the TVB time and data output vectors, and producing a list of scalar values. See the `sbi.demo.BoldFCDForSBI` for an example.

```python
from tvb_inversion.base.parameters import SimSeq

metrics = [BoldFCDForSBI(win_len=15)]
```

The simulations can be then performed using one of the helper functions, `utils.run_local` for local parallel execution or `utils.run_dask` for distributed execution in cluster environments.

```python
import utils

utils.run_local(seq, metrics, backend=NbMPRBackend, checkpoint_dir='test_run', filename='results.npy')
```

### Inference, diagnostics

When the simulations finish, we can continue with training the estimator and constructing the posterior. These steps are implemented in the `EstimatorSBI` class. For start, it loads the summary statistics for the sample.

```python
from tvb_inversion.sbi import EstimatorSBI

estimator = EstimatorSBI(stats_model=sbi_model, seq=seq, metrics=metrics)
summ_stats = estimator.load_summary_stats('results.npy')
```

Next, it trains the estimator returning the posterior object:

```python
posterior = estimator.train(summ_stats)
```

Given the summary statistics `obs_stats` computed from some empirical data, we can sample from the posterior with the following:

```python
num_samples = 20_000
posterior_samples = posterior.sample((num_samples,), obs_stats)
```

To assess the resulting distribution, we can compute the posterior shrinkage, that is how much the data is telling us about the parameters of interest (this value would be ideally close to 1 indicating well identified posterior):

```python
from tvb_inversion.sbi.inference import shrinkage, zscore

post_std = torch.std(posterior_samples)
post_mean = torch.mean(posterior_samples)

shr = shrinkage(prior_std, post_std)
```

On simulated data, we can compute the z-score over the posterior to assess how accurate is the posterior distribution with respect to the ground-truth:

```python
zsc = zscore(true_param, post_mean, post_std)
```



## Acknowledgments

This project has received funding from the European Union’s Horizon 2020 
Framework Programme for Research and Innovation under the Specific Grant 
Agreement No. 826421 - VirtualBrainCloud.
