import os
import time
import warnings

import matplotlib.pyplot as plt
import numba
import numpy as np
import sbi
import sbi.utils as utils
import sbi_julia
import torch
from julia.api import Julia
Julia(compiled_modules=False)
from julia import Main
from matplotlib import cm
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sbi.inference.base import infer
from scipy.stats import kurtosis
from scipy.stats import moment
from scipy.stats import skew
from sklearn import linear_model
from sbi_julia import extract_FCD

warnings.simplefilter("ignore")

print('Running on numpy: v{}'.format(np.version.version))
print('Running on torch: v{}'.format(torch.__version__))
print('Running on sbi: v{}'.format(sbi.__version__))

cwd = os.getcwd()
Res_dir = 'Res_SBI_MPR_aged_inferAlphaG_numsim1k'

print("Load MPR.jl")
MPR = Main.include("MPR.jl")

print("Load SC and masks")
sbi_julia_path = sbi_julia.__path__
SC_healthy = np.loadtxt(sbi_julia_path + '/data_input_files/SCjulichmax.txt', delimiter=',', unpack=True)
MaskJulichA = np.loadtxt(sbi_julia_path + '/data_input_files/MaskJulichA.txt', delimiter=',', unpack=True)
MaskJulichB = np.loadtxt(sbi_julia_path + '/data_input_files/MaskJulichB.txt', delimiter=',', unpack=True)
alpha = 0.5
beta = 0.0


def add_colorbar(fig, ax, im=None, vmin=None, vmax=None, position='right', size='1%', pad=0.3, **cbar_kwargs):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(position, size=size, pad=pad)
    if im is None:
        norm = colors.Normalize()
        im = plt.cm.ScalarMappable(norm=norm)
    if vmin is not None or vmax is not None:
        im.set_clim(vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, cax=cax, **cbar_kwargs)
    return cbar


def calculate_summary_statistics(x, nn, features):
    """Calculate summary statistics

    Parameters
    ----------
    x : output of the simulator

    Returns
    -------
    np.array, summary statistics
    """

    X = x.reshape(nn, int(x.shape[0] / nn))

    n_summary = 16 * nn + (nn * nn) + 300 * 300

    wwidth = 30
    maxNwindows = 200
    olap = 0.96

    sum_stats_vec = np.concatenate((np.mean(X, axis=1),
                                    np.median(X, axis=1),
                                    np.std(X, axis=1),
                                    skew(X, axis=1),
                                    kurtosis(X, axis=1),
                                    ))

    for item in features:

        if item == 'higher_moments':
            sum_stats_vec = np.concatenate((sum_stats_vec,
                                            moment(X, moment=2, axis=1),
                                            moment(X, moment=3, axis=1),
                                            moment(X, moment=4, axis=1),
                                            ))

        if item == 'FC_corr':
            FC = np.corrcoef(X)
            off_diag_sum_FC = np.sum(FC) - np.trace(FC)
            # eigen_vals_FC, _ = LA.eig(FC)
            # pca = PCA(n_components=3)
            # PCA_FC = pca.fit_transform(FC)

            sum_stats_vec = np.concatenate((sum_stats_vec,
                                            np.array([off_diag_sum_FC]),
                                            ))

        if item == 'FCD_corr':
            FCDcorr, Pcorr, shift = extract_FCD(X, wwidth, maxNwindows, olap, mode='corr')

            off_diag_sum_FCD = np.sum(FCDcorr) - np.trace(FCDcorr)
            # eigen_vals_FCD, _ = LA.eig(FCDcorr)
            # pca = PCA(n_components=3)
            # PCA_FCD = pca.fit_transform(FCDcorr)

            sum_stats_vec = np.concatenate((sum_stats_vec,
                                            np.array([off_diag_sum_FCD]),
                                            ))

    sum_stats_vec = sum_stats_vec[0:n_summary]

    return sum_stats_vec


def MPR_simulator_wrapper(params):
    params = np.asarray(params)

    params_alpha = params[0]
    params_G = params[1]

    SC_aged_ = SC_healthy - (params_alpha * MaskJulichA) - (beta * MaskJulichB)

    V_sim, R_sim = MPR(h, h_Store, sim_len, nCoeff, SC_aged_, ti, tf, J, delta, params_G, eta, skip_, count_)

    BOLD_v_sim = V_sim[::ds, :].T

    summstats = torch.as_tensor(
        calculate_summary_statistics(BOLD_v_sim.reshape(-1), nn, features=['higher_moments', 'FC_corr', 'FCD_corr']))

    return summstats


if __name__ == '__main__':
    print(SC_healthy.shape, MaskJulichA.shape, MaskJulichB.shape)
    SC_aged = SC_healthy - (alpha * MaskJulichA) - (beta * MaskJulichB)
    print("SC AGED: ", SC_aged.shape, SC_aged.min(), SC_aged.max())

    fig, axs = plt.subplots(ncols=4, figsize=(12, 8))

    ax = axs[0]
    im = ax.imshow(MaskJulichA)
    add_colorbar(fig, ax, im, size='4%', pad=0.1)
    ax.set(title='mask inter', ylabel='ROI', xlabel='ROI')

    ax = axs[1]
    im = ax.imshow(MaskJulichB)
    add_colorbar(fig, ax, im, size='4%', pad=0.1)
    ax.set(title='mask frontal', ylabel='ROI', xlabel='ROI')

    ax = axs[2]
    im = ax.imshow(SC_healthy)
    add_colorbar(fig, ax, im, size='4%', pad=0.1)
    ax.set(title='SC original', ylabel='ROI', xlabel='ROI')

    ax = axs[3]
    im = ax.imshow(SC_aged)
    add_colorbar(fig, ax, im, size='4%', pad=0.1)
    ax.set(title='SC aged', ylabel='ROI', xlabel='ROI')

    fig.tight_layout()

    plt.savefig(os.path.join(cwd + '/' + str(Res_dir), "SCdata.png"))

    fig, axs = plt.subplots(ncols=4, figsize=(12, 8))

    ax = axs[0]
    im = ax.imshow(np.log10(MaskJulichA))
    add_colorbar(fig, ax, im, size='4%', pad=0.1)
    ax.set(title='mask inter', ylabel='ROI', xlabel='ROI')

    ax = axs[1]
    im = ax.imshow(np.log10(MaskJulichB))
    add_colorbar(fig, ax, im, size='4%', pad=0.1)
    ax.set(title='mask frontal', ylabel='ROI', xlabel='ROI')

    ax = axs[2]
    im = ax.imshow(np.log10(SC_healthy))
    add_colorbar(fig, ax, im, size='4%', pad=0.1)
    ax.set(title='SC original', ylabel='ROI', xlabel='ROI')

    ax = axs[3]
    im = ax.imshow(np.log10(SC_aged))
    add_colorbar(fig, ax, im, size='4%', pad=0.1)
    ax.set(title='SC aged', ylabel='ROI', xlabel='ROI')

    fig.tight_layout()

    plt.savefig(os.path.join(cwd + '/' + str(Res_dir), "SCdata_log.png"))

    # simulation setup
    sim_len = 5  # 5 min simulation
    sim_len = sim_len * 60  # convert to sec
    input_freq = 10
    ti = 0
    skip_ = 10  # skip initial simulated time-points
    h_Store = 0.05
    h = 0.005  # dt of MPR
    tf = sim_len * (2 * input_freq)
    tf = (tf + skip_)
    count_ = int(2 * input_freq * h_Store / h)  # downsampling rate in the function.
    delta = 0.7
    eta = -4.6
    J = 14.5
    tau = 1
    nCoeff = 8  # usually it's integer. depending on the connectome, range is between 4 to 10
    delta = 0.7
    eta = -4.6
    J = 14.5
    tau = 1
    G = .84

    ## sanity check
    start_time = time.time()
    V, R = MPR(h, h_Store, sim_len, nCoeff, SC_aged, ti, tf, J, delta, G, eta, skip_, count_)
    print(" single sim (sec) takes:", (time.time() - start_time))
    print(V.shape, R.shape)

    ds = 20

    BOLD_v = V[::ds, :].T
    BOLD_r = R[::ds, :].T
    print(BOLD_v.shape, BOLD_r.shape)

    nt = BOLD_v.shape[1]
    nn = BOLD_v.shape[0]
    print(nn, nt)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(BOLD_v, aspect='auto', interpolation='none')
    ax.set(xlabel='time [s]', ylabel='ROI', title='BOLD_V Aged ')
    add_colorbar(fig, ax, im)
    plt.savefig(os.path.join(cwd + '/' + str(Res_dir), "Imshow_BOLD_aged_v.png"))

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(BOLD_r, aspect='auto', interpolation='none')
    ax.set(xlabel='time [s]', ylabel='ROI', title='BOLD_R Aged ')
    add_colorbar(fig, ax, im)
    plt.savefig(os.path.join(cwd + '/' + str(Res_dir), "Imshow_BOLD_aged_r.png"))

    plt.figure(figsize=(15, 4))
    plt.plot(BOLD_v.T)
    plt.xlabel("Time")
    plt.ylabel("V")
    plt.title("Simulated BOLD V signals Aged", fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(cwd + '/' + str(Res_dir), "SimulatedBOLDdata_v_aged.png"))
    plt.show()

    plt.figure(figsize=(15, 4))
    plt.plot(BOLD_r.T)
    plt.xlabel("Time")
    plt.ylabel("V")
    plt.title("Simulated BOLD R signals Aged", fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(cwd + '/' + str(Res_dir), "SimulatedBOLDdata_r_aged.png"))
    plt.show()

    FC_aged = np.corrcoef(BOLD_v)
    print(FC_aged.shape)

    plt.figure(figsize=(14, 6))
    plt.subplot(121)
    plt.imshow((SC_aged), interpolation='nearest', cmap='jet')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title('SC aged', fontsize=18)
    plt.subplot(122)
    im = plt.imshow(FC_aged, cmap=cm.jet)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.gca().set_title('Simulated FC aged', fontsize=18.0)
    plt.savefig(os.path.join(cwd + '/' + str(Res_dir), "SC_FC_aged.png"))
    plt.tight_layout(pad=2)
    plt.show()

    print(BOLD_v.shape)

    lenseries = len(BOLD_v[1])
    wwidth = 30
    maxNwindows = 200
    olap = 0.96
    Nwindows = min(((lenseries - wwidth * olap) // (wwidth * (1 - olap)), maxNwindows))
    shift = int((lenseries - wwidth) // (Nwindows - 1))
    if Nwindows == maxNwindows:
        wwidth = int(shift // (1 - olap))
    print(Nwindows, shift)

    indx_start = range(0, (lenseries - wwidth + 1), shift)
    indx_stop = range(wwidth, (1 + lenseries), shift)
    PcorrFCD_sim, Pcorr_sim, shift_sim = extract_FCD(BOLD_v, wwidth, maxNwindows, olap, mode='corr')

    fig, ax = plt.subplots(figsize=(6, 6))
    plt.imshow(PcorrFCD_sim, vmin=0, vmax=1, interpolation='nearest', cmap='jet')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(' Simulated FCD aged ', fontsize=14)
    # plt.axis([0, 250, 250, 0])
    plt.tight_layout()
    plt.savefig(os.path.join(cwd + '/' + str(Res_dir), "FCDSimulatedBOLD_v_aged.png"))
    plt.show()

    off_diag_sum_FC = np.sum(FC_aged) - np.trace(FC_aged)

    off_diag_sum_FCD = np.sum(PcorrFCD_sim) - np.trace(PcorrFCD_sim)

    calculate_summary_statistics = numba.jit(calculate_summary_statistics)

    _ = calculate_summary_statistics(BOLD_v.reshape(-1), nn, features=['higher_moments', 'FC_corr', 'FCD_corr'])
    print(_.shape)

    prior_min_alpha = 0.
    prior_min_G = 0 * np.ones(1)

    prior_max_alpha = 1.
    prior_max_G = 1.5 * np.ones(1)
    prior_min = np.hstack([prior_min_alpha, prior_min_G])
    prior_max = np.hstack([prior_max_alpha, prior_max_G])
    print(prior_min.shape, prior_max.shape)
    prior = utils.torchutils.BoxUniform(low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max))

    start_time = time.time()

    posterior = infer(MPR_simulator_wrapper, prior, method='SNPE', num_simulations=1000, num_workers=1)

    print("-" * 60)
    print("--- %s seconds ---" % (time.time() - start_time))

    ### Data #1
    alpha_true = 0.5
    beta_true = 0.0
    G_true = 0.84
    labels_params = [r'$\alpha$', r'$G$']
    params_true = np.hstack([alpha_true, G_true])

    SC_aged_true = SC_healthy - (alpha_true * MaskJulichA) - (beta_true * MaskJulichB)
    V_true, R_true = MPR(h, h_Store, sim_len, nCoeff, SC_aged_true, ti, tf, J, delta, G_true, eta, skip_, count_)
    BOLD_obs = V_true[::ds, :].T
    obs_summary_statistics = calculate_summary_statistics(BOLD_obs.reshape(-1), nn,
                                                          features=['higher_moments', 'FC_corr', 'FCD_corr'])
    print(BOLD_obs.shape, obs_summary_statistics.shape)
    plt.figure(figsize=(12, 3))
    plt.plot(BOLD_obs.T)
    plt.title("Simulated BOLD", fontsize=18)
    plt.xlabel('Time', fontsize=18)
    plt.ylabel('Amp', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(Res_dir, "SimulatedBOLD_obs1.png"))
    plt.show()
    FC_obs = np.corrcoef(BOLD_obs)
    print(FC_obs.shape)

    plt.figure(figsize=(14, 6))
    plt.subplot(121)
    plt.imshow((SC_aged_true), interpolation='nearest', cmap='jet')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title('SC aged', fontsize=18)
    plt.subplot(122)
    im = plt.imshow(FC_obs, cmap=cm.jet)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.gca().set_title('Simulated FC aged', fontsize=18.0)
    plt.savefig(os.path.join(Res_dir, "Sim_BOLD_FC_obs1.png"))
    plt.show()

    FCDcorr_obs, Pcorr_obs, shift_obs = extract_FCD(BOLD_obs, wwidth, maxNwindows, olap, mode='corr')
    plt.figure(figsize=(6, 6))
    plt.imshow(FCDcorr_obs, vmin=0, vmax=1, interpolation='nearest', cmap='jet')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title('FCD Pearson correlation', fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(Res_dir, "Sim_BOLD_FCD_obs1.png"))
    plt.show()

    ### Posterior Data #1
    num_samples = 1000
    posterior_samples = posterior.sample((num_samples,), obs_summary_statistics, sample_with_mcmc=True, ).numpy()
    print(posterior_samples.shape, params_true.shape)

    np.save(os.path.join(Res_dir, 'posterior_samples_MPR_aged_obs1.npy'), posterior_samples)

    alpha_posterior = posterior_samples[:, 0]
    G_posterior = posterior_samples[:, 1]

    print(alpha_true, G_true)
    print(alpha_posterior.mean(), G_posterior.mean())

    plt.figure(figsize=(4, 4))
    parts = plt.violinplot(alpha_posterior, widths=0.7, showmeans=True, showextrema=True)
    plt.plot(1, params_true[0], 'o', color='k', alpha=0.9, markersize=8)
    plt.ylabel(' Posterior ' + r'${(\alpha)}$', fontsize=18)
    plt.xlabel(r'${\alpha}$', fontsize=18)
    plt.xticks([])
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(cwd + '/' + str(Res_dir), "SBI_MPR_aged_InferredAlpha_obs1.png"), doi=800)
    plt.savefig(os.path.join(cwd + '/' + str(Res_dir), "SBI_MPR_aged_InferredAlpha_obs1.eps"), doi=800)
    plt.show()

    plt.figure(figsize=(4, 4))
    parts = plt.violinplot(G_posterior, widths=0.7, showmeans=True, showextrema=True)
    plt.plot(1, params_true[1], 'o', color='k', alpha=0.9, markersize=8)
    plt.ylabel(' Posterior ' + r'${(G)}$', fontsize=18)
    plt.xlabel(r'${G}$', fontsize=18)
    plt.xticks([])
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(cwd + '/' + str(Res_dir), "SBI_MPR_aged_InferredG_obs1.png"), doi=800)
    plt.savefig(os.path.join(cwd + '/' + str(Res_dir), "SBI_MPR_aged_InferredG_obs1.eps"), doi=800)
    plt.show()

    fig, axes = utils.pairplot(posterior_samples,
                               fig_size=(8, 8),
                               labels=labels_params,
                               upper=['kde'],
                               diag=['kde'],
                               points=params_true,
                               points_offdiag={'markersize': 20},
                               points_colors='r')
    plt.tight_layout()
    plt.savefig(os.path.join(cwd + '/' + str(Res_dir), "Posterior_pairplot_kde_obs_AlphaG_obs1.png"), doi=800)
    plt.savefig(os.path.join(cwd + '/' + str(Res_dir), "Posterior_pairplot_kde_obs_AlphaG_0bs1.eps"), doi=800)
    plt.show()

    plt.figure(figsize=(4, 4))
    plt.plot(G_posterior, alpha_posterior, '.')
    plt.plot(params_true[1], params_true[0], 'o', color='r', alpha=0.9, markersize=8)
    plt.xlabel(' Posterior ' + r'${(G)}$', fontsize=18)
    plt.ylabel(' Posterior ' + r'${(\alpha)}$', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(cwd + '/' + str(Res_dir), "SBI_MPR_aged_InferredAlphaG_obs1.png"), doi=800)
    plt.savefig(os.path.join(cwd + '/' + str(Res_dir), "SBI_MPR_aged_InferredAlphaG_obs1.eps"), doi=800)
    plt.show()

    print(G_posterior.shape)

    reg = linear_model.LinearRegression()
    reg.fit(G_posterior.reshape((-1, 1)), alpha_posterior.reshape((-1, 1)))
    m = reg.coef_[0]
    b = reg.intercept_
    print("slope=", m, "intercept=", b)
    predicted_alpha = reg.predict(G_posterior.reshape((-1, 1)))
    plt.figure(figsize=(4, 4))
    plt.plot(G_posterior, alpha_posterior, '.')
    plt.plot(G_posterior, predicted_alpha, 'cyan')
    plt.plot(params_true[1], params_true[0], 'o', color='r', alpha=0.9, markersize=8)
    plt.xlabel(' Posterior ' + r'${(G)}$', fontsize=18)
    plt.ylabel(' Posterior ' + r'${(\alpha)}$', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(cwd + '/' + str(Res_dir), "SBI_MPR_aged_LinregInferredAlphaG_obs1.png"), doi=800)
    plt.savefig(os.path.join(cwd + '/' + str(Res_dir), "SBI_MPR_aged_LinregInferredAlphaG_obs1.eps"), doi=800)
    plt.show()
    SC_aged_posterior = SC_healthy - (alpha_posterior.mean() * MaskJulichA) - (beta_true * MaskJulichB)
    V_fit, R_fit = MPR(h, h_Store, sim_len, nCoeff, SC_aged_posterior, ti, tf, J, delta, G_posterior.mean(), eta, skip_,
                       count_)
    BOLD_fit = V_fit[::ds, :].T
    np.save(os.path.join(Res_dir, 'BOLD_obs_MPR_aged_obs1.npy'), BOLD_obs)
    np.save(os.path.join(Res_dir, 'BOLD_fit_MPR_aged_obs1.npy'), BOLD_fit)

    plt.figure(figsize=(14, 4))
    plt.subplot(121)
    plt.plot(BOLD_obs.T)
    plt.xlabel("Time", fontsize=18)
    plt.ylabel("Amp", fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(" Observed BOLD data", fontsize=18)

    plt.subplot(122)
    plt.plot(BOLD_fit.T)
    plt.xlabel("Time", fontsize=18)
    plt.ylabel("Amp", fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(" Predicted  BOLD data", fontsize=18)
    plt.savefig(os.path.join(cwd + '/' + str(Res_dir), "SBI_MPR_aged_FittedSeriesobservation_obs1.png"), doi=800)
    plt.savefig(os.path.join(cwd + '/' + str(Res_dir), "SBI_MPR_aged_FittedSeriesobservation_obs1.eps"), doi=800)
    plt.show()

    plt.figure(figsize=(14, 4))
    plt.subplot(121)
    plt.imshow(BOLD_obs, aspect='auto', interpolation='bilinear', origin='lower', cmap='Reds')
    plt.colorbar()
    plt.xlabel("Time", fontsize=18)
    plt.ylabel("Regions#", fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(" Observed BOLD data", fontsize=18)

    plt.subplot(122)
    plt.imshow(BOLD_fit, aspect='auto', interpolation='bilinear', origin='lower', cmap='Reds')
    plt.colorbar()
    plt.xlabel("Time", fontsize=18)
    plt.ylabel("Regions#", fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(" Predicted BOLD data", fontsize=18)
    plt.savefig(os.path.join(cwd + '/' + str(Res_dir), "SBI_MPR_aged_ImshowFittedobservation_obs1.png"), doi=800)
    plt.savefig(os.path.join(cwd + '/' + str(Res_dir), "SBI_MPR_aged_ImshowFittedobservation_obs1.eps"), doi=800)
    plt.show()

    FC_obs = np.corrcoef(BOLD_obs)
    FC_fit = np.corrcoef(BOLD_fit)

    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow((FC_obs), interpolation='nearest', cmap='jet')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.gca().set_title('Observed FC', fontsize=18.0)
    plt.tight_layout()
    plt.subplot(122)
    im = plt.imshow(FC_fit, cmap=cm.jet)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.gca().set_title('Predicted FC', fontsize=18.0)
    plt.tight_layout()
    plt.savefig(os.path.join(Res_dir, "SBI_GenericHopf_SimvsPred_FC_obs1.png"), doi=800)
    plt.savefig(os.path.join(Res_dir, "SBI_GenericHopf_SimvsPred_FC_obs1.eps"), doi=800)
    plt.show()

    FCDcorr_obs, Pcorr_obs, shift_obs = extract_FCD(BOLD_obs, wwidth, maxNwindows, olap, mode='corr')
    FCDcorr_fit, Pcorr_fit, shift_fit = extract_FCD(BOLD_fit, wwidth, maxNwindows, olap, mode='corr')

    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(FCDcorr_obs, vmin=0, vmax=1, interpolation='nearest', cmap='jet')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.gca().set_title('Observed FCD', fontsize=18.0)
    plt.tight_layout()
    plt.subplot(122)
    plt.imshow(FCDcorr_fit, vmin=0, vmax=1, interpolation='nearest', cmap='jet')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.gca().set_title('Predicted FCD', fontsize=18.0)
    plt.tight_layout()
    plt.savefig(os.path.join(Res_dir, "SBI_GenericHopf_SimvsPred_FCD_obs1.png"), doi=800)
    plt.savefig(os.path.join(Res_dir, "SBI_GenericHopf_SimvsPred_FCD_obs1.eps"), doi=800)
    plt.show()
