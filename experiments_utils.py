import numpy as np
import torch
import time

import gb

import tsvar

# Fix numpy error behavior after gb, ignore underflow for softmax computations
np.seterr(under='ignore')

# Set numpy print format
np.set_printoptions(precision=2, floatmode='fixed', sign=' ')

MLE_N_ITER = 10000
BBVI_N_ITER = 20000
VI_FB_N_ITER = 5000
VI_N_ITER = 5000
GB_N_ITER = 300


def generate_parameters(dim, p=None, seed=None):
    """Generate a random adjacency matrix """
    if seed:
        np.random.seed(seed)
    # Default edge probability (Erdos-Renyi in conected regime)
    if p is None:
        p = 2 * np.log(dim) / dim
    # Sample baseline rates
    baseline = np.random.uniform(0.0, 0.1, size=dim).round(4)
    # Sample beta
    beta = np.random.uniform(0.0, 2.0, size=(dim, dim)).round(4)
    # Sample adjacency (itera until a stable process is found)
    for _ in range(10):
        adjacency = np.random.binomial(n=1, p=p, size=(dim, dim))
        adjacency = adjacency.astype(float)
        adjacency *= np.random.uniform(0.1, 0.2, size=(dim, dim))
        adjacency = adjacency.round(4)
        # Check stability
        wold_sim = tsvar.simulate.MultivariateWoldSimulator(
            mu_a=baseline, alpha_ba=adjacency, beta_ba=beta)
        if wold_sim.spectral_radius() < 1:
            return {'baseline': baseline.tolist(),
                    'beta': beta.tolist(),
                    'adjacency': adjacency.tolist()}
    raise RuntimeError("Could not generate stable process. Gave up...")


def generate_data(baseline, beta, adjacency, max_jumps, sim_seed=None):
    """Generate a realization of a multivariate Wold process given parameters"""
    # Set random seed (for reproducibility)
    if sim_seed is None:
        sim_seed = np.random.randint(2**31 - 1)
    # Simulate a realization
    wold_sim = tsvar.simulate.MultivariateWoldSimulator(
        mu_a=baseline, alpha_ba=adjacency, beta_ba=beta)
    events = wold_sim.simulate(max_jumps=max_jumps, seed=sim_seed)
    events = [torch.tensor(ev, dtype=torch.float) for ev in events]
    end_time = wold_sim.end_time
    # Ensure observations in every dimension
    assert min(map(len, events)) > 0
    return events, end_time, sim_seed


def coeffs_array_to_dict(coeffs_hat, n_base, n_beta, n_adj):
    """Convert array of coefficient into dict of parameter types"""
    coeffs_dict = {}
    dim = n_base
    assert len(coeffs_hat) == n_base + n_beta + n_adj
    # baseline
    coeffs_dict['baseline'] = coeffs_hat[:n_base].tolist()
    # beta (only if present)
    if n_beta > 0:
        coeffs_dict['beta'] = np.array(
            coeffs_hat[n_base:n_base+n_beta]).reshape(dim, dim).tolist()
    # adjacency
    coeffs_dict['adjacency'] = np.array(
        coeffs_hat[n_base+n_beta:]).reshape(dim, dim).tolist()
    return coeffs_dict


def run_mle(events, end_time, coeffs_true_dict, seed):
    dim = len(events)
    # set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Set initial guess
    coeffs_start = torch.tensor(np.hstack((
        np.random.uniform(0.0, 1.0, size=dim),     # baseline
        np.random.uniform(0.0, 1.0, size=dim**2),  # beta
        np.random.uniform(0.0, 1.0, size=dim**2)   # adjacency
    )))
    # Extract ground truth
    coeffs_true = np.hstack((coeffs_true_dict['baseline'],
                             np.array(coeffs_true_dict['beta']).flatten(),
                             np.array(coeffs_true_dict['adjacency']).flatten()))
    # Define model
    model = tsvar.models.WoldModelMLE(verbose=True)
    model.observe(events, end_time)
    # Set callback
    callback = tsvar.utils.callbacks.LearnerCallbackMLE(
        coeffs_start, print_every=500, coeffs_true=coeffs_true,
        acc_thresh=0.05, dim=dim, default_end="\n")
    # Fit model
    conv = model.fit(x0=coeffs_start, optimizer=torch.optim.Adam, lr=0.05,
                     lr_sched=0.9999, tol=1e-5, max_iter=MLE_N_ITER,
                     penalty=tsvar.priors.GaussianPrior, C=1e10,
                     seed=None, callback=callback)
    coeffs_hat = model.coeffs.detach().numpy()
    # Print results
    print('\nConverged?', conv)
    max_diff = np.max(np.abs(coeffs_true - coeffs_hat))
    print(f'  - coeffs_hat:  {coeffs_hat.round(2)}')
    print(f'  - coeffs_true: {coeffs_true.round(2)}')
    print(f'  - max_diff: {max_diff:.4f}')
    # Save result
    res_dict = {}
    res_dict['coeffs'] = coeffs_array_to_dict(coeffs_hat, n_base=dim,
                                              n_beta=dim**2, n_adj=dim**2)
    res_dict['conv'] = conv
    res_dict['history'] = callback.to_dict()
    return res_dict


def run_bbvi(events, end_time, coeffs_true_dict, seed):
    dim = len(events)
    n_params = dim + dim**2 + dim**2
    # set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Set initial guess
    coeffs_start = torch.tensor(np.hstack((
        # loc
        -2.0 * torch.ones(dim, dtype=torch.float),                  # baseline
        0.0 * torch.ones((dim, dim), dtype=torch.float).flatten(),  # beta
        0.0 * torch.ones((dim, dim), dtype=torch.float).flatten(),  # adjacency
        # scale
        torch.log(0.2 * torch.ones(dim, dtype=torch.float)),
        torch.log(0.2 * torch.ones((dim, dim), dtype=torch.float).flatten()),
        torch.log(0.2 * torch.ones((dim, dim), dtype=torch.float).flatten()),
    )))
    # Extract ground truth
    coeffs_true = np.hstack((coeffs_true_dict['baseline'],
                             np.array(coeffs_true_dict['beta']).flatten(),
                             np.array(coeffs_true_dict['adjacency']).flatten()))
    # Define priors/posteriors
    posterior = tsvar.posteriors.LogNormalPosterior
    prior = tsvar.priors.GaussianLaplacianPrior
    mask_gaus = torch.zeros(n_params, dtype=torch.bool)
    mask_gaus[:dim + dim**2] = 1  # Gaussian prior for baseline and beta
    C = 1e3
    # Init the model object
    model = tsvar.models.WoldModelBBVI(posterior=posterior, prior=prior, C=C,
                                       prior_kwargs={'mask_gaus': mask_gaus},
                                       n_samples=1, n_weights=1, weight_temp=1,
                                       verbose=False, device='cpu')
    model.observe(events, end_time)

    # Set link function for callback (vi coeffs -> posterior mode)
    def link_func(coeffs):
        """variationa coeffs -> posterior mode of adjacency"""
        # Numpy to torch
        coeffs = torch.tensor(coeffs) if isinstance(coeffs, np.ndarray) else coeffs
        return model.posterior.mode(
            coeffs[:model.n_params], coeffs[model.n_params:]
        ).detach().numpy()[dim+dim**2:]
    # Set the callback (callback parameters are posterior mode)
    callback = tsvar.utils.callbacks.LearnerCallbackMLE(
        x0=posterior().mode(
            coeffs_start[:dim+2*dim**2], coeffs_start[dim+2*dim**2:]
        )[dim+dim**2:],
        print_every=500,
        coeffs_true=coeffs_true,
        acc_thresh=0.05,
        dim=dim,
        link_func=link_func,
        default_end="\n")
    # Fit the model
    conv = model.fit(x0=coeffs_start, optimizer=torch.optim.Adam, lr=1e-1,
                     lr_sched=0.9999, tol=1e-6, max_iter=BBVI_N_ITER,
                     mstep_interval=100, mstep_offset=500, mstep_momentum=0.5,
                     seed=None, callback=callback)
    # Print results
    print('\nConverged?', conv)
    coeffs_hat_mean = model.posterior.mean(model.coeffs[:model.n_params],
                                           model.coeffs[model.n_params:]
                                           ).detach().numpy().round(2)
    coeffs_hat_mode = model.posterior.mode(model.coeffs[:model.n_params],
                                           model.coeffs[model.n_params:]
                                           ).detach().numpy().round(2)
    max_diff_mean = np.max(np.abs(coeffs_true - coeffs_hat_mean))
    max_diff_mode = np.max(np.abs(coeffs_true - coeffs_hat_mode))
    print(f'  - coeffs_hat_mean:  {coeffs_hat_mean.round(2)}')
    print(f'  - coeffs_hat_mode:  {coeffs_hat_mode.round(2)}')
    print(f'  - coeffs_true:      {coeffs_true.round(2)}')
    print(f'  - max_diff_mean: {max_diff_mean:.4f}')
    print(f'  - max_diff_mode: {max_diff_mode:.4f}')
    # Save result
    res_dict = {}
    res_dict['coeffs'] = {
        'loc': model.coeffs[:model.n_params].tolist(),
        'log-scale': model.coeffs[model.n_params:].tolist()
    }
    res_dict['conv'] = conv
    res_dict['history'] = callback.to_dict()
    return res_dict


def run_vi_fixed_beta(events, end_time, coeffs_true_dict, seed):
    dim = len(events)
    # Extract ground truth
    coeffs_true = np.hstack((coeffs_true_dict['baseline'],
                             coeffs_true_dict['adjacency'].flatten()))
    # Set model
    model = tsvar.models.WoldModelVariationalFixedBeta(verbose=True)
    model.observe(events, beta=coeffs_true_dict['beta'])
    # Set priors
    as_pr = 1.0 * np.ones((dim + 1, dim))
    ar_pr = 1.0 * np.ones((dim + 1, dim))
    zc_pr = [1.0 * np.ones((len(events[i]), dim+1)) for i in range(dim)]
    # Set callback (parameters of callback are just the posterior mean of alpha)
    callback = tsvar.utils.callbacks.LearnerCallbackMLE(
        x0=(as_pr / ar_pr).flatten(), print_every=500,
        coeffs_true=coeffs_true_dict['adjacency'].flatten(),
        acc_thresh=0.05, dim=dim, default_end="\n")
    # Fit model
    conv = model.fit(as_pr=as_pr, ar_pr=ar_pr, zc_pr=zc_pr, max_iter=VI_FB_N_ITER,
                     tol=1e-5, callback=callback)
    # Print results
    print('\nConverged?', conv)
    coeffs_hat_mean = model.alpha_posterior_mean().flatten()
    coeffs_hat_mode = model.alpha_posterior_mode().flatten()
    max_diff_mean = np.max(np.abs(coeffs_true - coeffs_hat_mean))
    max_diff_mode = np.max(np.abs(coeffs_true - coeffs_hat_mode))
    print(f'  - coeffs_hat_mean:  {coeffs_hat_mean.round(2)}')
    print(f'  - coeffs_hat_mode:  {coeffs_hat_mode.round(2)}')
    print(f'  - coeffs_true:      {coeffs_true.round(2)}')
    print(f'  - max_diff_mean: {max_diff_mean:.4f}')
    print(f'  - max_diff_mode: {max_diff_mode:.4f}')
    # Save result
    res_dict = {}
    res_dict['coeffs'] = {
        'as_po': model._as_po.tolist(),
        'ar_po': model._ar_po.tolist(),
        'beta': coeffs_true_dict['beta'].tolist(),
    }
    res_dict['conv'] = conv
    res_dict['history'] = callback.to_dict()
    return res_dict


def run_vi(events, end_time, coeffs_true_dict, seed):
    dim = len(events)
    # Set model
    model = tsvar.models.WoldModelVariational(verbose=True)
    model.observe(events)
    # Set priors
    # prior: Alpha
    as_pr = 1.0 * np.ones((dim + 1, dim))
    ar_pr = 1.0 * np.ones((dim + 1, dim))
    # prior: Beta
    bs_pr = 100.0 * np.ones((dim, dim))
    br_pr = 100.0 * np.ones((dim, dim))
    # prior: Z
    zc_pr = [1.0 * np.ones((len(events[i]), dim+1)) for i in range(dim)]
    # Extract ground truth (for callback, only alphas)
    coeffs_true = np.hstack((
        coeffs_true_dict['baseline'],
        coeffs_true_dict['adjacency'].flatten()))
    # Set callback (parameters of callback are just the posterior mean of alpha)
    callback = tsvar.utils.callbacks.LearnerCallbackMLE(
        x0=(as_pr / ar_pr).flatten(), print_every=500, coeffs_true=coeffs_true,
        acc_thresh=0.05, dim=dim, default_end='\n')
    # Fit model
    conv = model.fit(as_pr=as_pr, ar_pr=ar_pr, bs_pr=bs_pr, br_pr=br_pr,
                     zc_pr=zc_pr, max_iter=VI_N_ITER, tol=1e-5, callback=callback)
    # Print results
    print('\nConverged?', conv)
    coeffs_hat_mean = model.alpha_posterior_mean().flatten()
    coeffs_hat_mode = model.alpha_posterior_mode().flatten()
    max_diff_mean = np.max(np.abs(coeffs_true - coeffs_hat_mean))
    max_diff_mode = np.max(np.abs(coeffs_true - coeffs_hat_mode))
    print(f'  - coeffs_hat_mean:  {coeffs_hat_mean.round(2)}')
    print(f'  - coeffs_hat_mode:  {coeffs_hat_mode.round(2)}')
    print(f'  - coeffs_true:      {coeffs_true.round(2)}')
    print(f'  - max_diff_mean: {max_diff_mean:.4f}')
    print(f'  - max_diff_mode: {max_diff_mode:.4f}')
    # Save result
    res_dict = {}
    res_dict['coeffs'] = {
        'as_po': model._as_po.tolist(),
        'ar_po': model._ar_po.tolist(),
        'bs_po': model._bs_po.tolist(),
        'br_po': model._br_po.tolist(),
    }
    res_dict['conv'] = conv
    res_dict['history'] = callback.to_dict()
    return res_dict


def run_gb(events, end_time, coeffs_true_dict, seed):
    dim = len(events)
    # Set random seed
    np.random.seed(seed)
    # Define model
    granger_model = gb.GrangerBusca(
        alpha_prior=1.0/len(events),
        num_iter=GB_N_ITER,
        metropolis=True,
        beta_strategy='busca',
    )  # recommended parameters
    start_time = time.time()
    granger_model.fit(events)
    run_time = time.time() - start_time
    # Extract infered adjacency
    adj_hat = granger_model.Alpha_.toarray()
    adj_hat = adj_hat / adj_hat.sum(axis=0)
    beta_hat = np.ones((dim, dim)) * (granger_model.beta_ + 1)
    coeffs_hat = np.hstack((granger_model.mu_, beta_hat.flatten(),
                            adj_hat.flatten()))
    # Extract ground truth
    coeffs_true = np.hstack((coeffs_true_dict['baseline'],
                             coeffs_true_dict['beta'].flatten(),
                             coeffs_true_dict['adjacency'].flatten()))
    max_diff = np.max(np.abs(coeffs_true - coeffs_hat))
    print(f'  - coeffs_hat:  {coeffs_hat.round(2)}')
    print(f'  - coeffs_true: {coeffs_true.round(2)}')
    print(f'  - max_diff: {max_diff:.4f}')
    # Save result
    res_dict = {}
    res_dict['coeffs'] = {
        'baseline': granger_model.mu_.tolist(),
        'beta': beta_hat.tolist(),
        'adjacency': adj_hat.tolist()
    }
    res_dict['conv'] = True
    res_dict['history'] = {'time': [run_time]}
    return res_dict