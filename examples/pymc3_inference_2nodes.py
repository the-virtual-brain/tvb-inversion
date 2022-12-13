from tvb_inversion.pymc3.examples import (
    create_2node_simulator,
    default_model_builders,
    uninformative_model_builders,
    custom_model_builders
)


if __name__ == "__main__":
    sim = create_2node_simulator(simulation_length=250)
    (t, X), = sim.run()

    data, summary = default_model_builders(sim=sim, observation=X, draws=250, tune=250, cores=2, target_accept=0.9, max_treedepth=15)
    # data, summary = uninformative_model_builders(sim=sim, observation=X, draws=250, tune=250, cores=2, target_accept=0.9, max_treedepth=15)
    # data, summary = custom_model_builders(sim=sim, observation=X, draws=250, tune=250, cores=2, target_accept=0.9, max_treedepth=15)
