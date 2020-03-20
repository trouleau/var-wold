import numpy as np
import torch

import tsvar


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
    beta = torch.tensor([1.0, 2.0])
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
    print('  - Test succeeded!')


def generate_test_dataset():
    print('  - Define model parameters...')
    # Define random parameters
    dim = 2  # Dimensionality of the process
    end_time = 2e5  # Choose a long observation window
    mu = torch.tensor([0.3, 0.1])
    beta = torch.tensor([1.11, 1.5])
    alpha = torch.tensor([
        [0.7, 0.3],
        [0.0, 1.0]
    ])
    coeffs_true = torch.cat((mu, beta, alpha.flatten())).numpy()
    print('  - Simulate lots of data...')
    np.random.seed(42)  # Fix random seed for data generation
    # Simulate lots of data
    wold_sim = tsvar.simulate.GrangerBuscaSimulator(
        mu_rates=mu, Alpha_ba=alpha, Beta_b=beta)
    events = wold_sim.simulate(end_time, seed=4243)
    events = [torch.tensor(ev, dtype=torch.float) for ev in events]
    print((f"    - Simulated {sum(map(len, events)):,d} events "
           f"with end time: {end_time}"))
    return dim, coeffs_true, events, end_time


def test_wold_model_mle():
    print('\nTesting: WoldModel MLE:')

    dim, coeffs_true, events, end_time = generate_test_dataset()

    print('  - Run MLE...')
    # Define model
    model = tsvar.models.WoldModel(verbose=True)
    model.observe(events, end_time)
    coeffs_start = torch.cat((
        0.5 * torch.ones(dim, dtype=torch.float),
        2.0 * torch.ones(dim, dtype=torch.float),
        0.5 * torch.ones((dim, dim), dtype=torch.float).flatten()
    ))

    callback = tsvar.utils.callbacks.LearnerCallbackMLE(
        coeffs_start, coeffs_true, print_every=10)

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
    assert max_diff < 0.1, 'Test FAILED !!!'
    print('  - Test succeeded! (max_diff < 0.1)')


def test_wold_model_bbvi():
    print('\nTesting: WoldModel BBVI:')

    dim, coeffs_true, events, end_time = generate_test_dataset()

    print('  - Run BBVI...')
    posterior = tsvar.posteriors.LogNormalPosterior
    prior = tsvar.priors.GaussianPrior
    C = 1000.0

    model = tsvar.models.WoldModelBBVI(posterior=posterior, prior=prior, C=C,
                                       n_samples=1, n_weights=1, weight_temp=1,
                                       verbose=False, device='cpu')
    model.observe(events, end_time)

    coeffs_start = torch.cat((
        0.5 * torch.ones(dim, dtype=torch.float),
        2.0 * torch.ones(dim, dtype=torch.float),
        0.5 * torch.ones((dim, dim), dtype=torch.float).flatten()
    ))

    callback = tsvar.utils.callbacks.LearnerCallbackMLE(
        coeffs_start, coeffs_true, print_every=10)

    conv =  model.fit(x0=coeffs_start, optimizer=torch.optim.Adam, lr=0.05,
                      lr_sched=0.9999, tol=1e-5, max_iter=100000,
                      penalty=tsvar.priors.GaussianPrior, C=10000.0,
                      seed=None, callback=callback)


if __name__ == "__main__":
    # test_wold_model_loglikelihood()
    test_wold_model_mle()
    # test_wold_model_bbvi()
