from tvb_inversion.pymc.examples import (
    create_2node_simulator,
    default_model_builders,
    uninformative_model_builders,
    custom_model_builders
)


if __name__ == "__main__":
    sim = create_2node_simulator(simulation_length=250)
    (t, X), = sim.run()

    idata, isummary = default_model_builders(sim=sim, observation=X, draws=250, tune=250, cores=2, target_accept=0.9)
    # idata, isummary = uninformative_model_builders(sim=sim, observation=X, draws=250, tune=250, cores=2, target_accept=0.9)
    # idata, isummary = custom_model_builders(sim=sim, observation=X, draws=250, tune=250, cores=2, target_accept=0.9)
