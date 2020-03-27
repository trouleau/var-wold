import numpy as np
import torch

import tsvar
import gb


# Set numpy print format
np.set_printoptions(precision=2, floatmode='fixed', sign='+')


def test_wold_model_loglikelihood():
    print('\nTesting: WoldModel.log_likelihood:')

    # Toy events
    events = [
        torch.tensor([1.0, 3.0, 4.0, 5.0]),
        torch.tensor([2.0, 4.5, 6.0])
    ]
    # Toy end time of observation
    end_time = 10.0
    # Parameters of the model, to evaluate the log-likelihood
    mu = torch.tensor([0.2, 0.1])
    beta = torch.tensor([1.0, 2.0]) - 1.0
    A = torch.tensor([[0.1, 0.9],
                      [0.4, 0.6]])
    coeffs = torch.cat((mu, beta, A.flatten()))
    # Create an instance of `WoldModel`, set the data and evaluate the
    # log-likelihood
    model = tsvar.models.WoldModel()
    model.observe(events, end_time)
    ll_computed = model.log_likelihood(coeffs)

    # Define the groud-truth intensity at each observed event
    lam_true = [
        np.array([
            0.2,
            0.2,
            0.2 + 0.1 / (1 + 2) + 0.4 / (2 + 1),
            0.2 + 0.1 / (1 + 1) + 0.4 / (2 + 2)
        ]),
        np.array([
            0.1,
            0.1 + 0.9 / (1 + 1),
            0.1 + 0.9 / (1 + 0.5) + 0.6 / (2 + 2.5)
        ])
    ]
    # Define the groud-truth intensity at the end time
    lam_end_true = [
        0.2 + 0.1 / (1 + 1) + 0.4 / (2 + 0.5),
        0.1 + 0.9 / (1 + 1) + 0.6 / (2 + 1.5)
    ]
    # Compute the ground-truth log-likelood value
    ll_true = (
        np.log(lam_true[0]).sum()
        - np.sum(lam_true[0] * np.hstack((events[0][0], np.diff(events[0]))))
        - lam_end_true[0] * (end_time - events[0][-1])

        + np.log(lam_true[1]).sum()
        - np.sum(lam_true[1] * np.hstack((events[1][0], np.diff(events[1]))))
        - lam_end_true[1] * (end_time - events[1][-1])
    )
    print(f'  - Computed loglik.: {ll_computed.item():.5f}')
    print(f'  - Ground truth loglik.: {ll_true.item():.5f}')
    assert np.isclose(ll_computed, ll_true), 'Test FAILED !!!'
    print('  - Test SUCESS!')


def generate_test_dataset():
    print()
    print('Generate a toy dataset')
    print('----------------------')
    print('  - Define model parameters...')
    # Define random parameters
    dim = 2  # Dimensionality of the process
    end_time = 5e4  # Choose a long observation window
    mu = torch.tensor([0.3, 0.1])
    beta = torch.tensor([1.0, 0.2])
    # Use the same constraint as GrangerBusca to allow fair comparison
    alpha = torch.tensor([
        [0.7, 0.3],
        [0.0, 1.0]
    ])
    coeffs_true = torch.cat((mu, beta, alpha.flatten())).numpy()
    print('  - Simulate lots of data...')
    # Simulate lots of data
    wold_sim = tsvar.simulate.GrangerBuscaSimulator(
        mu_rates=mu, Alpha_ba=alpha, Beta_b=beta)
    events = wold_sim.simulate(end_time, seed=4243)
    events = [torch.tensor(ev, dtype=torch.float) for ev in events]
    print((f"    - Simulated {sum(map(len, events)):,d} events "
           f"with end time: {end_time}"))
    return dim, coeffs_true, events, end_time


def test_wold_model_mle(dim, coeffs_true, events, end_time):
    print()
    print('Testing: WoldModel MLE')
    print('----------------------')

    # Define model
    model = tsvar.models.WoldModelMLE(verbose=True)
    model.observe(events, end_time)
    coeffs_start = torch.cat((
        0.5 * torch.ones(dim, dtype=torch.float),
        1.0 * torch.ones(dim, dtype=torch.float),
        0.5 * torch.ones((dim, dim), dtype=torch.float).flatten()
    ))

    callback = tsvar.utils.callbacks.LearnerCallbackMLE(
        coeffs_start, print_every=10, coeffs_true=coeffs_true,
        acc_thresh=0.05, dim=dim)

    conv = model.fit(x0=coeffs_start, optimizer=torch.optim.Adam, lr=0.05,
                     lr_sched=0.9999, tol=1e-5, max_iter=1000,
                     penalty=tsvar.priors.GaussianPrior, C=1e10,
                     seed=None, callback=callback)
    coeffs_hat = model.coeffs.detach().numpy()

    print('\nConverged?', conv)
    max_diff = np.max(np.abs(coeffs_true - coeffs_hat))
    print(f'  - coeffs_hat:  {coeffs_hat.round(2)}')
    print(f'  - coeffs_true: {coeffs_true.round(2)}')
    print(f'  - max_diff: {max_diff:.4f}')
    if max_diff >= 0.1:
        print('  - Test FAILED !!!')
    else:
        print('  - Test SUCESS! (max_diff < 0.1)')


def test_wold_model_bbvi(dim, coeffs_true, events, end_time):
    print()
    print('Testing: WoldModel BBVI')
    print('-----------------------')

    posterior = tsvar.posteriors.LogNormalPosterior
    prior = tsvar.priors.GaussianLaplacianPrior
    mask_gaus = torch.zeros(coeffs_true.shape, dtype=torch.bool)
    mask_gaus[:2*dim] = 1
    C = 1e3
    # Init the model object
    model = tsvar.models.WoldModelBBVI(posterior=posterior, prior=prior, C=C,
                                       prior_kwargs={'mask_gaus': mask_gaus},
                                       n_samples=1, n_weights=1, weight_temp=1,
                                       verbose=False, device='cpu')
    model.observe(events, end_time)
    # Set initial guess for parameters
    coeffs_start = torch.cat((
        # loc
        0.2 * torch.ones(dim, dtype=torch.float),
        1.0 * torch.ones(dim, dtype=torch.float),
        0.5 * torch.ones((dim, dim), dtype=torch.float).flatten(),
        # scale
        torch.log(0.2 * torch.ones(dim, dtype=torch.float)),
        torch.log(0.2 * torch.ones(dim, dtype=torch.float)),
        torch.log(0.2 * torch.ones((dim, dim), dtype=torch.float).flatten()),
    ))
    # Set the callback object to print stuff and keep track of algorithm history
    callback = tsvar.utils.callbacks.LearnerCallbackMLE(
        coeffs_start, print_every=10)
    # Fit the model
    conv = model.fit(x0=coeffs_start, optimizer=torch.optim.Adam, lr=1e-1,
                     lr_sched=0.9999, tol=1e-6, max_iter=10000,
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
    if max_diff_mean >= 0.1 or max_diff_mode >= 0.1:
        print('  - Test FAILED !!!')
    else:
        print('  - Test SUCESS! (max_diff < 0.1)')


def test_granger_busca(dim, coeffs_true, events, end_time):
    """

    Note: This test never passed. Even if using `gb` own simulator and inference
    code.
    """
    print()
    print('Testing: Granger Busca')
    print('----------------------')
    # Define model
    granger_model = gb.GrangerBusca(
        alpha_prior=1.0/len(events),
        num_iter=300,
        metropolis=True,
        beta_strategy='busca',
    )  # recommended parameters
    granger_model.fit(events)
    # Extract infered adjacency
    adj_hat = granger_model.Alpha_.toarray()
    adj_hat = adj_hat / adj_hat.sum(axis=0)
    # Extract infered beta, here we give the advantage of the beta_j^i fixed
    # for all i
    beta_hat = np.ones((dim, dim)) * granger_model.beta_ + 1

    coeffs_hat = np.hstack((granger_model.mu_, beta_hat.flatten(), adj_hat.flatten()))
    max_diff = np.max(np.abs(coeffs_true - coeffs_hat))
    print(f'  - coeffs_hat:  {coeffs_hat.round(2)}')
    print(f'  - coeffs_true: {coeffs_true.round(2)}')
    print(f'  - max_diff: {max_diff:.4f}')
    if max_diff < 0.1:
        print('  - Test SUCESS! (max_diff < 0.1)')
    else:
        print('  - Test FAILED !!!')


if __name__ == "__main__":
    # test_wold_model_loglikelihood()
    # print('\n', '='*80, '\n', sep='')

    data = generate_test_dataset()

    test_wold_model_mle(*data)
    print('\n', '='*80, '\n', sep='')
    test_wold_model_bbvi(*data)
    print('\n', '='*80, '\n', sep='')
    test_granger_busca(*data)
