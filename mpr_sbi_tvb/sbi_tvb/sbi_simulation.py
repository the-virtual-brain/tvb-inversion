from typing import Callable, Any, Tuple

import torch
from joblib import Parallel, delayed
from sbi.inference import check_if_proposal_has_default_x
from sbi.simulators.simutils import tqdm_joblib
from torch import Tensor
from tqdm import tqdm


def simulate_in_batches(
        simulator: Callable,
        theta: Tensor,
        sim_batch_size: int = 1,
        num_workers: int = 1,
        show_progress_bars: bool = True,
) -> Tensor:
    r"""
    Return simulations $x$ for parameters $\theta$ conducted batchwise.

    Parameters are batched with size `sim_batch_size` (default whole theta at once).
    Multiprocessing is used when `num_workers > 1`.

    Args:
        simulator: Simulator callable (a function or a class with `__call__`).
        theta: All parameters $\theta$ sampled from prior or posterior.
        sim_batch_size: Number of simulations per batch. Default is to simulate
            the entire theta in a single batch. When using multiple workers, increasing
            this batch size can further speed up simulations by reducing overhead.
        num_workers: Number of workers for multiprocessing.
        show_progress_bars: Whether to show a progress bar during simulation.

    Returns:
        Parameters theta and simulations $x$.
    """

    num_sims, *_ = theta.shape

    if num_sims == 0:
        x = torch.tensor([])
    elif sim_batch_size is not None and sim_batch_size < num_sims:
        # Dev note: pyright complains of torch.split lacking a type stub
        # as of PyTorch 1.4.0, see https://github.com/microsoft/pyright/issues/291
        batches = torch.split(theta, sim_batch_size, dim=0)

        if num_workers > 1:
            # Parallelize the sequence of batches across workers.
            # We use the solution proposed here: https://stackoverflow.com/a/61689175
            # to update the pbar only after the workers finished a task.
            with tqdm_joblib(
                    tqdm(
                        batches,
                        disable=not show_progress_bars,
                        desc=f"Running {num_sims} simulations in {len(batches)} batches.",
                        total=len(batches),
                    )
            ) as progress_bar:
                simulation_outputs = Parallel(n_jobs=num_workers)(
                    delayed(simulator)(batch) for batch in batches
                )
        else:
            pbar = tqdm(
                total=num_sims,
                disable=not show_progress_bars,
                desc=f"Running {num_sims} simulations.",
            )

            with pbar:
                simulation_outputs = []
                for batch in batches:
                    simulation_outputs.append(simulator(batch))
                    pbar.update(sim_batch_size)

        x = torch.cat(simulation_outputs, dim=0)
    else:
        x = simulator(theta)

    return x


def simulate_for_sbi(
        simulator: Callable,
        proposal: Any,
        num_simulations: int,
        num_workers: int = 1,
        simulation_batch_size: int = 1,
        show_progress_bar: bool = True,
) -> Tuple[Tensor, Tensor]:
    r"""
    Returns ($\theta, x$) pairs obtained from sampling the proposal and simulating.

    This function performs two steps:

    - Sample parameters $\theta$ from the `proposal`.
    - Simulate these parameters to obtain $x$.

    Args:
        simulator: A function that takes parameters $\theta$ and maps them to
            simulations, or observations, `x`, $\text{sim}(\theta)\to x$. Any
            regular Python callable (i.e. function or class with `__call__` method)
            can be used.
        proposal: Probability distribution that the parameters $\theta$ are sampled
            from.
        num_simulations: Number of simulations that are run.
        num_workers: Number of parallel workers to use for simulations.
        simulation_batch_size: Number of parameter sets that the simulator
            maps to data x at once. If None, we simulate all parameter sets at the
            same time. If >= 1, the simulator has to process data of shape
            (simulation_batch_size, parameter_dimension).
        show_progress_bar: Whether to show a progress bar for simulating. This will not
            affect whether there will be a progressbar while drawing samples from the
            proposal.

    Returns: Sampled parameters $\theta$ and simulation-outputs $x$.
    """

    check_if_proposal_has_default_x(proposal)

    theta = proposal.sample((num_simulations,))

    x = simulate_in_batches(
        simulator, theta, simulation_batch_size, num_workers, show_progress_bar
    )

    return theta, x
