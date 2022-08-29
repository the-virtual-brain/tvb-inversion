# TVB INVERSION

Module capable to compute inference for TVB simulator parameters, based on the SBI library.

### API

The **TVBInference** class is central to this module. It receives a TVB simulator and a list of priors as arguments.
Then, it is able to sample these priors and compute a posterior estimation for a simulation parameter.

Therefore, the API consists of the following steps:

1. Configure a TVB simulator
2. Define priors
3. Sample priors
4. Train
5. Compute posterior


For step 2, the priors are defined using the **Prior** class by choosing the simulator parameter to infer and an interval to sample from:
   
   
    prior = Prior('coupling.a', 1.5, 3.2)

And then, the TVBInference object should be created by giving it the previously configured TVB simulator, a list of priors and a directory to store the results:

    
    tvb_inference = TvbInference(sim=tvb_simulator,
                                 priors=[prior],
                                 output_dir=output_dir)

The third step (priors sampling) can be executed with different backends, either locally or remotely to an HPC that
supports the UNICORE API. Only users with access to an HPC (eg. DAINT-CSCS) will be able to use the remote backend.

Each user should choose the preferred backend within this API:

- remotely (an HPC project and site have to be chosen by instantiating the UnicoreConfig object)


    unicore_config = UnicoreConfig(project='hpc_project', site='DAINT-CSCS')
    tvb_inference.sample_priors(num_simulations=10, 
                                num_workers=10, 
                                backend=BackendEnum.REMOTE,
                                unicore_config=unicore_config)

- locally


    tvb_inference.sample_priors(num_simulations=10,
                                num_workers=10)


The fourth step can also be accessed from TVBInference object and can run locally:

    tvb_inference.train()

And finally, the last step to compute posterior estimation receives as argument the observation data (empirical or simulated time series):
    
    posterior_samples = tvb_inference.posterior(data=observation_data)


### Usage

Examples of usage are described in:

- this python [script](https://github.com/the-virtual-brain/tvb-inversion/blob/main/mpr_sbi_tvb/sim_inference_refactored.py)
- or these [notebooks](https://github.com/the-virtual-brain/tvb-inversion/tree/main/mpr_sbi_tvb/notebooks)

## Acknowledgments

This project has received funding from the European Union’s Horizon 2020 
Framework Programme for Research and Innovation under the Specific Grant 
Agreement No. 826421 - VirtualBrainCloud.
