# TVB INVERSION

Module capable to compute inference for TVB simulator parameters.

### API

The **TVBInference** class is central to this module. It receives a TVB simulator and a list of priors as arguments.
Then, it is able to sample these priors and compute a posterior estimation for a simulation parameter.

Therefore, the API consists of the following steps:

1. Configure a TVB simulator
2. Define priors
3. Sample priors
4. Train
5. Compute posterior

The third step (priors sampling) can be executed with different backends, either locally or remotely to an HPC that
supports the UNICORE API.

Each user should choose the preferred backend within this API:

- remotely 
  - only for users with access to an HPC (DAINT-CSCS for now)
  - an HPC project has to be given as argument


    tvb_inference = TvbInference(...)
    tvb_inference.sample_priors(num_simulations=10, num_workers=10, backend=BackendEnum.REMOTE, project='hpc_project')

- locally


    tvb_inference = TvbInference(...)
    tvb_inference.sample_priors(num_simulations=10, num_workers=10)

### Usage

Examples of usage are described in:

- this python [script](https://github.com/the-virtual-brain/tvb-inversion/blob/main/mpr_sbi_tvb/sim_inference_refactored.py)
- or these [notebooks](https://github.com/the-virtual-brain/tvb-inversion/tree/main/mpr_sbi_tvb/notebooks)

## Acknowledgments

This project has received funding from the European Union’s Horizon 2020 
Framework Programme for Research and Innovation under the Specific Grant 
Agreement No. 826421 - VirtualBrainCloud.
